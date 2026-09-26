# ============================================================================
# dualbranch_unet.py
# ----------------------------------------------------------------------------
# 목적:
#   MRI(전체) + 병변영역(MRI⊙mask) 두 입력을 받아 PET을 합성하는 BBDM용
#   denoiser UNet. downsample 모듈만 3종(M1/M2/M3) 갈아끼워 비교하기 위한
#   실험용 백본이다.
#
# 왜 dual-branch 인가:
#   - x1_t = bridge 상태 (t=0에서 PET, t=T에서 MRI로 보간된 볼륨)  → Branch A
#   - x2   = y2 = MRI ⊙ mask (병변 영역만 남긴 볼륨, 보조 조건)     → Branch B
#   BBDM 특성상 y1(전체 MRI)은 bridge 수식 안에 이미 녹아 있으므로
#   (x_t = (1-m)x0 + m·y1 + noise) 네트워크에 별도로 안 넣는다.
#   네트워크가 추가로 받는 조건은 y2(병변)뿐이다. 이게 일반 conditional
#   diffusion(=MRI를 concat)과의 핵심 차이.
#
# 비교 변수(오직 이것만 바꿈):
#   Down 모듈 = {maxpool(M1), swin(M2), zeroconv(M3)}
#   나머지(채널/해상도/attention/ResBlock/upsample/skip)는 세 모델 100% 동일.
#   → downsample 방식이 "병변 정보 보존"에 주는 효과만 순수 분리하기 위함.
#
# 해상도/채널 스케줄 (base=16 기준):
#   A: 128³(16) → 64³(32) → 32³(64)
#   B: 128³(16) → 64³(32) → 32³(64)
#   F(fusion): concat→32³(128) → 16³(128)
#   M(bottleneck): 16³(128)
#   Decoder: 16³→32³→64³→128³, B branch의 skip을 각 해상도에서 concat
#
# attention:
#   전부 3D window attention(4³ = 64 token)만 쓴다. global(전체 token)은
#   32³ 이상에서 바로 OOM 나는 걸 이미 겪었으므로(einsum weight = N²),
#   어느 해상도든 싼 4³ window로 통일한다. window가 작아 전역 관계는 약하지만
#   shift로 window 간 정보를 섞어 부분 보완한다.
#   ※ 세 모델(M1/M2/M3)에서 attention 설정은 반드시 동일해야 한다.
#     M2(swin merging)에만 window attention을 더 붙이면 "downsample 효과"가
#     아니라 "Swin encoder 효과"를 재는 꼴이 되어 비교가 오염된다.
# ============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 공통 유틸
# ----------------------------------------------------------------------------
def GN(c):
    """GroupNorm. group 수는 min(8, c).
    주의: GroupNorm(g, c)는 c가 g로 나눠떨어져야 한다. base=16이면 최소 채널이
    16이라 group=8로 항상 안전(16%8=0). 채널을 8 미만으로 내리지 말 것."""
    return nn.GroupNorm(min(8, c), c)


def sinusoidal(t, dim=64):
    """diffusion timestep t(정수 텐서, shape [B])를 sinusoidal 임베딩으로.
    Transformer positional encoding과 동일한 sin/cos 방식.
    반환 shape: [B, dim]."""
    half = dim // 2
    # 주파수: 10000^(-i/half), i=0..half-1  → 저주파~고주파
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]          # [B, half]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)  # [B, dim]


# ----------------------------------------------------------------------------
# ResBlock (timestep 임베딩 주입, AdaGN-lite 방식)
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    """3D residual block.
    구조:
      x → GN → SiLU → Conv3x3 → (+ temb) → GN → SiLU → Conv3x3 → (+ skip)
    temb는 채널 방향으로 더해지는 bias 형태로 주입(간단하고 안정적).
    c_in != c_out 이면 skip 경로에 1x1 conv로 채널을 맞춘다.
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.n1 = GN(min(8, c_in))
        self.c1 = nn.Conv3d(c_in, c_out, 3, padding=1)
        # temb(256차원)를 c_out 채널 bias로 사영
        self.emb = nn.Linear(256, c_out)
        self.n2 = GN(min(8, c_out))
        self.c2 = nn.Conv3d(c_out, c_out, 3, padding=1)
        # 채널 수가 바뀌면 skip도 1x1로 맞춤, 아니면 그대로 통과
        self.skip = nn.Conv3d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        # temb 주입: [B,c_out] → [B,c_out,1,1,1] 로 broadcast
        h = h + self.emb(F.silu(temb))[:, :, None, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


# ----------------------------------------------------------------------------
# 4³ Window Attention (3D), SDPA 사용
# ----------------------------------------------------------------------------
class WindowAttn3D(nn.Module):
    """3D window self-attention.
    - 볼륨을 4×4×4 window로 쪼개고, 각 window(=64 token) 안에서만 attention.
    - window가 작아(64 token) N²이 4096으로 고정 → 어느 해상도든 값싸다.
      (global attention은 N=전체 voxel이라 32³=32768 token → N²=10^9, OOM.)
    - shift: 레이어마다 window 경계를 half(=2)만큼 굴려(roll) window 사이
      정보가 섞이게 한다(Swin의 shifted window와 동일 아이디어).
    - attention 연산은 F.scaled_dot_product_attention(SDPA)로.
      SDPA는 memory-efficient/flash backend가 잡히면 attention matrix를
      명시적으로 만들지 않아 메모리가 안정적이다. (직접 einsum으로 N×N을
      만들면 여기서 OOM 났었음.)

    입력/출력 shape: [B, C, D, H, W] 동일 (해상도/채널 불변, residual add).
    제약: D,H,W 가 각각 ws(=4)로 나눠떨어져야 한다.
          우리 스케줄(128/64/32/16)은 전부 4의 배수라 OK.
    """
    def __init__(self, c, ws=4, heads=4, shift=0):
        super().__init__()
        self.ws = ws
        self.heads = heads
        self.shift = shift            # 0 또는 ws//2(=2). 레이어마다 번갈아.
        self.norm = GN(min(8, c))
        self.qkv = nn.Conv3d(c, c * 3, 1)   # 1x1 conv로 q,k,v 한번에
        self.proj = nn.Conv3d(c, c, 1)      # 출력 사영

    def forward(self, x):
        B, C, D, H, W = x.shape
        ws = self.ws
        h = self.norm(x)

        # shifted window: 경계를 굴려서 다음 레이어가 다른 조합을 보게 함
        if self.shift:
            h = torch.roll(h, shifts=(-self.shift, -self.shift, -self.shift),
                           dims=(2, 3, 4))

        qkv = self.qkv(h)   # [B, 3C, D, H, W]

        # ---- window 분할 ----
        # [B, 3C, D,H,W] → head/채널/window 축으로 재배열
        # 최종 목표: (num_window, heads, 64, c_per_head) 형태로 SDPA에 투입
        ch = C // self.heads
        qkv = qkv.reshape(B, 3, self.heads, ch,
                          D // ws, ws, H // ws, ws, W // ws, ws)
        # 축 순서: B, nD, nH, nW, 3, heads, ch, wsD, wsH, wsW
        qkv = qkv.permute(0, 4, 6, 8, 1, 2, 3, 5, 7, 9).contiguous()
        # (B*nD*nH*nW) 를 배치로, (ws^3=64) 를 token 으로
        qkv = qkv.reshape(-1, 3, self.heads, ch, ws * ws * ws)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]   # 각 (nwin, heads, ch, 64)

        # SDPA는 (..., seq, dim) 규약 → (nwin, heads, 64, ch) 로 transpose
        q = q.transpose(-1, -2)   # (nwin, heads, 64, ch)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        o = F.scaled_dot_product_attention(q, k, v)   # (nwin, heads, 64, ch)

        # ---- window 복원 (분할의 역순) ----
        o = o.transpose(-1, -2).reshape(
            B, D // ws, H // ws, W // ws, self.heads, ch, ws, ws, ws)
        o = o.permute(0, 4, 5, 1, 6, 2, 7, 3, 8).contiguous()
        o = o.reshape(B, C, D, H, W)
        o = self.proj(o)

        # shift 되돌리기
        if self.shift:
            o = torch.roll(o, shifts=(self.shift, self.shift, self.shift),
                           dims=(2, 3, 4))

        return x + o   # residual


# ============================================================================
# Down 모듈 3종 (이 실험의 유일한 변수)
#   인터페이스 통일: Down(c_in, c_out)(x, temb) -> [B, c_out, D/2, H/2, W/2]
#   temb는 M3(zeroconv)만 실제로 사용. M1/M2는 받되 무시(인터페이스 통일용).
# ============================================================================
class DownMaxpool(nn.Module):
    """M1 — MaxPool 방식.
      MaxPool3d(2)로 공간 1/2 축소(각 2×2×2 블록의 최댓값) 후 3x3 conv로 채널 변환.
    특징: 파라미터 최소, 학습 안 하는 고정 다운샘플. 최댓값만 남겨 3/4 정보는
          버린다 → 병변처럼 '작고 값이 중요한' 영역은 손실 위험.
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = nn.Conv3d(c_in, c_out, 3, padding=1)

    def forward(self, x, temb=None):
        return self.conv(self.pool(x))


class DownSwin(nn.Module):
    """M2 — Swin patch merging (3D).
      2×2×2 이웃 8개를 '버리지 않고' 채널 축으로 concat(→ 8·c_in 채널, 해상도 1/2)
      한 뒤, LayerNorm → 1x1 conv 로 c_out 채널로 축소.
    특징: 공간 정보를 전혀 버리지 않는 유일한 변형(다 채널로 이동). 정보 보존
          관점에서 가장 유리할 것으로 기대. 1x1 conv로 채널만 다룸.
    주의: 이웃을 뽑는 (i,j,k) 순서는 아래처럼 고정하고 문서화해야 재현된다.
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.ln = nn.LayerNorm(8 * c_in)   # 채널 축 LayerNorm (Swin 방식)
        self.reduce = nn.Conv3d(8 * c_in, c_out, 1)

    def forward(self, x, temb=None):
        # (i,j,k) ∈ {0,1}^3, 순서 고정: i(깊이) → j(높이) → k(너비)
        parts = [x[:, :, i::2, j::2, k::2]
                 for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        x8 = torch.cat(parts, dim=1)               # [B, 8·c_in, D/2, H/2, W/2]
        # LayerNorm은 마지막 축 기준 → 채널을 마지막으로 옮겼다가 되돌림
        h = self.ln(x8.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()
        return self.reduce(h)


class ZeroConv(nn.Module):
    """1x1 conv, weight/bias를 0으로 초기화.
    초기 출력이 0 → 학습 시작 시 side 경로가 main을 건드리지 않아 안정적으로
    출발하고, 학습이 진행되며 0에서부터 기여가 커진다(ControlNet 핵심 아이디어)."""
    def __init__(self, c):
        super().__init__()
        self.c = nn.Conv3d(c, c, 1)
        nn.init.zeros_(self.c.weight)
        nn.init.zeros_(self.c.bias)

    def forward(self, x):
        return self.c(x)


class DownZeroconv(nn.Module):
    """M3 — ControlNet zero-conv (병렬 버전).
      main 경로 : AvgPool(2) → 1x1 conv  (항상 열려 있어 처음부터 신호가 흐름)
      side 경로 : Conv(k=2,s=2)로 다운샘플 → ResBlock(temb 주입) → ZeroConv
      out = main + ZeroConv(side)
    특징: 초기엔 side가 0이라 main(AvgPool)만 작동 → 안정. 학습하며 side가
          점점 기여. temb를 유일하게 쓰는 다운샘플(시간 조건부 변형).
    주의: '직렬' 버전(Conv(k2s2) 뒤에 ZeroConv만)은 초기 출력이 전부 0이라
          수렴이 크게 느려진다. 반드시 이 '병렬' 버전을 메인으로 쓸 것.
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        # main: 평균 풀링(부드럽게 축소) + 채널 변환
        self.pool = nn.AvgPool3d(2)
        self.main = nn.Conv3d(c_in, c_out, 1)
        # side: 학습되는 다운샘플 경로
        self.side_down = nn.Conv3d(c_in, c_in, 2, stride=2)  # 공간 1/2
        self.side_rb = ResBlock(c_in, c_out)
        self.zero = ZeroConv(c_out)                          # 초기 0

    def forward(self, x, temb):
        main = self.main(self.pool(x))
        side = self.side_rb(self.side_down(x), temb)
        return main + self.zero(side)


def build_down(kind, c_in, c_out):
    """다운샘플 팩토리. 모델 본체는 이 함수 인자만 바꿔 세 모델을 만든다.
    kind ∈ {'maxpool','swin','zeroconv'}."""
    table = {'maxpool': DownMaxpool, 'swin': DownSwin, 'zeroconv': DownZeroconv}
    if kind not in table:
        raise ValueError(f"unknown down_kind: {kind}")
    return table[kind](c_in, c_out)


# ----------------------------------------------------------------------------
# Upsample (세 모델 공통, 고정)
# ----------------------------------------------------------------------------
class Up(nn.Module):
    """nearest 보간으로 공간 2배 → 3x3 conv.
    ※ 3D 전체 축(D,H,W)을 모두 2배로 키운다. (원본 BBDM openaimodel의
       Up/Downsample은 3D에서 D축을 안 건드리는 버그가 있었다 — 여기선
       처음부터 등방으로 구현해 그 문제를 원천 차단.)"""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv3d(c, c, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # D,H,W 모두 2배
        return self.conv(x)


# ============================================================================
# Dual-branch UNet 본체
# ============================================================================
class DualBranchUNet(nn.Module):
    """
    입력:
      x1 : bridge 상태 x_t          [B,1,128,128,128]
      x2 : 병변영역 y2 = MRI⊙mask   [B,1,128,128,128]
      timesteps : diffusion t       [B]  (정수)
    출력:
      objective 예측                [B,1,128,128,128]

    down_kind 로 M1/M2/M3 를 고른다. use_attn_hi=True면 고해상도(128³,64³)
    슬롯에도 4³ window attention을 켠다(OOM 안 나므로 켜도 됨). 세 모델에서
    반드시 같은 값을 쓸 것.
    """
    def __init__(self, down_kind='maxpool', base=16, use_attn_hi=True):
        super().__init__()
        self.down_kind = down_kind

        # timestep 임베딩 MLP: sinusoidal(64) → 256 → 256
        self.temb = nn.Sequential(
            nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 256))

        D = lambda ci, co: build_down(down_kind, ci, co)  # 짧게 쓰기 위한 alias
        # shift는 attention 레이어마다 0 / ws//2(=2) 번갈아 쓰는 게 이상적이지만
        # 여기선 슬롯이 한 곳당 1개라 단순히 shift=2로 고정(원하면 확장).
        A = lambda c: WindowAttn3D(c, shift=2)

        # ---------- Branch A : x1(bridge) ----------
        self.A0 = nn.Conv3d(1, base, 3, padding=1)                 # 128³, base
        self.A1 = nn.ModuleList([ResBlock(base, base), ResBlock(base, base)])
        self.attn_A128 = A(base) if use_attn_hi else nn.Identity()
        self.A2 = D(base, base * 2)                                # →64³, 2base
        self.A3 = nn.ModuleList([ResBlock(base * 2, base * 2),
                                 ResBlock(base * 2, base * 2)])
        self.attn_A64 = A(base * 2) if use_attn_hi else nn.Identity()
        self.A4 = D(base * 2, base * 4)                            # →32³, 4base
        self.A5 = nn.ModuleList([ResBlock(base * 4, base * 4),
                                 ResBlock(base * 4, base * 4)])
        self.attn_A32 = A(base * 4)                                # 32³ attention

        # ---------- Branch B : x2(병변) ----------
        # B는 skip을 각 해상도에서 뽑아 decoder로 넘긴다(S128/S64/S32).
        self.B0 = nn.Conv3d(1, base, 3, padding=1)                 # 128³ → S128
        self.B1 = D(base, base * 2)                                # →64³
        self.B2 = nn.ModuleList([ResBlock(base * 2, base * 2),
                                 ResBlock(base * 2, base * 2)])
        self.attn_B64 = A(base * 2) if use_attn_hi else nn.Identity()  # →S64
        self.B3 = D(base * 2, base * 4)                            # →32³
        self.B4 = nn.ModuleList([ResBlock(base * 4, base * 4),
                                 ResBlock(base * 4, base * 4)])
        self.attn_B32 = A(base * 4)                                # →S32

        # ---------- Fusion (A5 + B4) ----------
        self.F1 = nn.Conv3d(base * 8, base * 4, 3, padding=1)      # concat(8base)→4base
        self.F2 = nn.ModuleList([ResBlock(base * 4, base * 8),
                                 ResBlock(base * 8, base * 8)])     # →8base
        self.attn_F32 = A(base * 8)
        self.F3 = D(base * 8, base * 8)                            # →16³, 8base

        # ---------- Bottleneck (16³) ----------
        self.M = nn.ModuleList([ResBlock(base * 8, base * 8) for _ in range(4)])
        self.attn_M = A(base * 8)                                  # 16³ attention (항상 on)

        # ---------- Decoder : upsample 후 B skip concat ----------
        self.U3 = Up(base * 8)                                     # 16→32
        self.D3 = nn.ModuleList([ResBlock(base * 8 + base * 4, base * 4),  # +S32
                                 ResBlock(base * 4, base * 4)])
        self.U2 = Up(base * 4)                                     # 32→64
        self.D2 = nn.ModuleList([ResBlock(base * 4 + base * 2, base * 2),  # +S64
                                 ResBlock(base * 2, base * 2)])
        self.U1 = Up(base * 2)                                     # 64→128
        self.D1 = nn.ModuleList([ResBlock(base * 2 + base, base),          # +S128
                                 ResBlock(base, base)])
        self.out = nn.Sequential(GN(min(8, base)), nn.SiLU(),
                                 nn.Conv3d(base, 1, 3, padding=1))  # →[B,1,128³]

    def forward(self, x1, x2, timesteps, **kwargs):
        # **kwargs: BBDM 코어가 context=None, y=clinical 등을 넘겨도 무시하기 위함
        temb = self.temb(sinusoidal(timesteps))   # [B,256]

        # ----- Branch A -----
        a = self.A0(x1)
        for r in self.A1:
            a = r(a, temb)
        a = self.attn_A128(a)
        a = self.A2(a, temb)
        for r in self.A3:
            a = r(a, temb)
        a = self.attn_A64(a)
        a = self.A4(a, temb)
        for r in self.A5:
            a = r(a, temb)
        a = self.attn_A32(a)             # [B,4base,32³]

        # ----- Branch B (skip 저장) -----
        b = self.B0(x2)
        S128 = b                          # skip @128
        b = self.B1(b, temb)
        for r in self.B2:
            b = r(b, temb)
        b = self.attn_B64(b)
        S64 = b                           # skip @64
        b = self.B3(b, temb)
        for r in self.B4:
            b = r(b, temb)
        b = self.attn_B32(b)
        S32 = b                           # skip @32  [B,4base,32³]

        # ----- Fusion -----
        f = torch.cat([a, b], dim=1)      # [B,8base,32³]
        f = self.F1(f)                    # →4base
        for r in self.F2:
            f = r(f, temb)                # →8base
        f = self.attn_F32(f)
        f = self.F3(f, temb)              # →16³, 8base

        # ----- Bottleneck -----
        for r in self.M:
            f = r(f, temb)
        f = self.attn_M(f)                # [B,8base,16³]

        # ----- Decoder (B skip concat) -----
        h = self.U3(f)                    # 16→32
        h = torch.cat([h, S32], dim=1)    # +S32
        for r in self.D3:
            h = r(h, temb)
        h = self.U2(h)                    # 32→64
        h = torch.cat([h, S64], dim=1)    # +S64
        for r in self.D2:
            h = r(h, temb)
        h = self.U1(h)                    # 64→128
        h = torch.cat([h, S128], dim=1)   # +S128
        for r in self.D1:
            h = r(h, temb)
        return self.out(h)                # [B,1,128³]


# ----------------------------------------------------------------------------
# 단독 실행 시 shape 자가 점검 (64³ 축소본으로 빠르게)
#   실제 학습 전에 python dualbranch_unet.py 로 구조가 도는지만 확인.
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    for kind in ['maxpool', 'swin', 'zeroconv']:
        net = DualBranchUNet(down_kind=kind, base=16, use_attn_hi=True)
        n = sum(p.numel() for p in net.parameters()) / 1e6
        # 64³ 축소본(메모리 절약) — 스케줄이 4의 배수라 64도 통과
        x1 = torch.randn(1, 1, 64, 64, 64)
        x2 = torch.randn(1, 1, 64, 64, 64)
        t = torch.randint(0, 1000, (1,))
        y = net(x1, x2, t)
        print(f"[{kind}] params={n:.2f}M  out={tuple(y.shape)}")
