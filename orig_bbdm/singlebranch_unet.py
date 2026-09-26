# ============================================================================
# singlebranch_unet.py
# ----------------------------------------------------------------------------
# 목적:
#   dual-branch 모델의 "비교 기준선(baseline)".
#   pixel space BBDM에서 MRI⊙mask(병변 영역 MRI)만을 bridge 종점(y)으로 쓰고,
#   네트워크 입력은 x_t 하나뿐인 단일 branch UNet.
#
# dual-branch와의 관계 (공정 비교를 위한 설계 원칙):
#   - 블록 구현(ResBlock / WindowAttn3D / Down / Up / temb)은 dualbranch_unet.py
#     에서 그대로 import 한다. → 코드 차이로 인한 성능 차이가 원천적으로 없음.
#   - 해상도/채널 스케줄 동일:
#       128³(16) → 64³(32) → 32³(64) → [F2: 32³(128)] → 16³(128) bottleneck
#       → 32³(64) → 64³(32) → 128³(16)
#   - attention 슬롯 동일 (4³ window attention):
#       A@128, A@64 (use_attn_hi=True일 때), A@32, F@32, M@16
#   - down 모듈 동일하게 선택 가능 (maxpool / swin / zeroconv)
#   - 차이는 오직 하나: B branch(병변 조건 경로)가 없다.
#       dual  : x_t(A) + y2(B) → fusion(concat) → decoder(A skip + B skip)
#       single: x_t(A)          → F1(채널만 맞춤) → decoder(A skip만)
#
# 왜 A skip을 반드시 넣나 (overfit 실패에서 얻은 교훈):
#   BBDM objective(grad) = m·(y−x0) + σ·ε = x_t − x0 이다.
#   즉 네트워크는 x_t를 거의 그대로 보존한 채 x0만 빼야 하는데, x_t의 noise
#   성분(σ·ε)은 고주파라 다운샘플을 거치면 사라진다. 인코더 쪽 skip이 없으면
#   decoder가 x_t를 복원하지 못해 noise 항을 못 맞추고 loss가 ~0.44에 고정된다.
#   → 표준 U-Net처럼 각 해상도의 인코더 feature를 skip으로 decoder에 넘긴다.
#
# 입력/출력:
#   x1 : x_t (bridge 상태) [B,1,128,128,128]
#   timesteps : [B] 정수
#   출력 : objective 예측 [B,1,128,128,128]
# ============================================================================

import torch
import torch.nn as nn

# dual-branch와 "완전히 같은" 블록을 쓰기 위해 직접 import
from dualbranch_unet import (ResBlock, WindowAttn3D, build_down, Up, GN,
                             sinusoidal)


class SingleBranchUNet(nn.Module):
    def __init__(self, down_kind='maxpool', base=16, use_attn_hi=True):
        super().__init__()
        self.down_kind = down_kind

        # timestep 임베딩 (dual-branch와 동일)
        self.temb = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 256))

        D = lambda ci, co: build_down(down_kind, ci, co)   # 다운샘플 팩토리
        A = lambda c: WindowAttn3D(c, shift=2)              # 4³ window attention

        # ---------- Encoder (= dual-branch의 Branch A와 동일) ----------
        self.A0 = nn.Conv3d(1, base, 3, padding=1)                     # 128³, base
        self.A1 = nn.ModuleList([ResBlock(base, base), ResBlock(base, base)])
        self.attn_A128 = A(base) if use_attn_hi else nn.Identity()
        self.A2 = D(base, base * 2)                                    # →64³
        self.A3 = nn.ModuleList([ResBlock(base * 2, base * 2),
                                 ResBlock(base * 2, base * 2)])
        self.attn_A64 = A(base * 2) if use_attn_hi else nn.Identity()
        self.A4 = D(base * 2, base * 4)                                # →32³
        self.A5 = nn.ModuleList([ResBlock(base * 4, base * 4),
                                 ResBlock(base * 4, base * 4)])
        self.attn_A32 = A(base * 4)

        # ---------- F (fusion 자리) ----------
        # dual에선 concat(A5,B4)=8base → 4base 였다.
        # single은 B가 없으므로 4base → 4base 로 같은 3x3 conv 한 층을 둔다.
        # (층 수/해상도/이후 채널을 dual과 맞추기 위함)
        self.F1 = nn.Conv3d(base * 4, base * 4, 3, padding=1)
        self.F2 = nn.ModuleList([ResBlock(base * 4, base * 8),
                                 ResBlock(base * 8, base * 8)])
        self.attn_F32 = A(base * 8)
        self.F3 = D(base * 8, base * 8)                                # →16³

        # ---------- Bottleneck (16³) ----------
        self.M = nn.ModuleList([ResBlock(base * 8, base * 8) for _ in range(4)])
        self.attn_M = A(base * 8)

        # ---------- Decoder (A skip만 concat) ----------
        self.U3 = Up(base * 8)                                         # 16→32
        self.D3 = nn.ModuleList([ResBlock(base * 8 + base * 4, base * 4),   # +SA32
                                 ResBlock(base * 4, base * 4)])
        self.U2 = Up(base * 4)                                         # 32→64
        self.D2 = nn.ModuleList([ResBlock(base * 4 + base * 2, base * 2),   # +SA64
                                 ResBlock(base * 2, base * 2)])
        self.U1 = Up(base * 2)                                         # 64→128
        self.D1 = nn.ModuleList([ResBlock(base * 2 + base, base),           # +SA128
                                 ResBlock(base, base)])
        self.out = nn.Sequential(GN(base), nn.SiLU(),
                                 nn.Conv3d(base, 1, 3, padding=1))

    def forward(self, x1, timesteps, **kwargs):
        temb = self.temb(sinusoidal(timesteps))

        # ----- Encoder (skip 저장) -----
        a = self.A0(x1)
        for r in self.A1:
            a = r(a, temb)
        a = self.attn_A128(a)
        SA128 = a
        a = self.A2(a, temb)
        for r in self.A3:
            a = r(a, temb)
        a = self.attn_A64(a)
        SA64 = a
        a = self.A4(a, temb)
        for r in self.A5:
            a = r(a, temb)
        a = self.attn_A32(a)
        SA32 = a

        # ----- F -----
        f = self.F1(a)
        for r in self.F2:
            f = r(f, temb)
        f = self.attn_F32(f)
        f = self.F3(f, temb)

        # ----- Bottleneck -----
        for r in self.M:
            f = r(f, temb)
        f = self.attn_M(f)

        # ----- Decoder -----
        h = self.U3(f)
        h = torch.cat([h, SA32], dim=1)
        for r in self.D3:
            h = r(h, temb)
        h = self.U2(h)
        h = torch.cat([h, SA64], dim=1)
        for r in self.D2:
            h = r(h, temb)
        h = self.U1(h)
        h = torch.cat([h, SA128], dim=1)
        for r in self.D1:
            h = r(h, temb)
        return self.out(h)


if __name__ == '__main__':
    # shape 자가 점검 (64³ 축소본)
    for kind in ['maxpool', 'swin', 'zeroconv']:
        net = SingleBranchUNet(down_kind=kind)
        n = sum(p.numel() for p in net.parameters()) / 1e6
        x = torch.randn(1, 1, 64, 64, 64)
        t = torch.randint(0, 1000, (1,))
        print(f"[single-{kind}] params={n:.2f}M out={tuple(net(x, t).shape)}")
