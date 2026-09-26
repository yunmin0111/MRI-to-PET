# ============================================================================
# our_dualbranch_bbdm.py
# ----------------------------------------------------------------------------
# 목적:
#   Brownian Bridge Diffusion Model(BBDM)을 pixel space에서, dual-branch UNet
#   (x_t + y2 병변조건)으로 돌리기 위한 모델 래퍼.
#
# 왜 상속이 아니라 자체 구현인가:
#   기존 xuekt98/BBDM의 BrownianBridgeModel은 denoise_fn(x_t) 입력이 1개다.
#   우리는 denoise_fn(x_t, y2) 로 입력이 2개라 p_losses/sample을 그대로 못 쓴다.
#   BBDM 코어를 건드리면 다른 실험(latent BBDM 등)에 영향이 가므로, 여기서는
#   BBDM 수식을 자체적으로(작게) 다시 구현한다. 수식/계수는 프로젝트에서
#   xuekt98/BBDM 구현과 1:1 대조해 이미 검증한 것을 그대로 옮겼다.
#
# BBDM 수식 (objective='grad', m_t=t/T 선형):
#   m_t      = t / T
#   var_t    = 2·s·(m_t − m_t²)          # s=max_var (기본 1.0). δ_t 와 동일.
#   x_t      = (1−m_t)·x0 + m_t·y1 + sqrt(var_t)·ε
#   target   = m_t·(y1 − x0) + sqrt(var_t)·ε      # 네트워크가 맞출 objective
#   loss     = L1( f_θ(x_t, y2, t),  target )
#
#   ※ y1(전체 MRI)은 x_t 안에 구조적으로 들어가므로 네트워크에 따로 안 넣는다.
#     네트워크가 받는 추가 조건은 y2(병변)뿐. 이게 일반 conditional diffusion과
#     의 결정적 차이이며 논문에서 명확히 구분해야 하는 지점.
#
# 샘플링(역과정)은 프로젝트 context section 10의 검증된 DDIM-류 계수를 사용.
# ============================================================================

import numpy as np
import torch
import torch.nn as nn

from dualbranch_unet import DualBranchUNet


class OurDualBranchBBDM(nn.Module):
    """
    dual-branch BBDM (pixel space).

    Args:
      down_kind   : 'maxpool' | 'swin' | 'zeroconv'  (비교 대상 다운샘플)
      base        : UNet 기본 채널 (기본 16)
      use_attn_hi : 고해상도 슬롯에도 4³ window attention 켤지 (세 모델 동일해야 함)
      num_timesteps : 학습 timestep 수 T (기본 1000)
      sample_step : 샘플링 step 수 (기본 200)
      max_var     : δ_t 스케일 s (기본 1.0)
    """
    def __init__(self, down_kind='maxpool', base=16, use_attn_hi=True,
                 num_timesteps=1000, sample_step=200, max_var=1.0):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.sample_step = sample_step
        self.max_var = max_var
        self.eta = 1.0   # 샘플링 확률성(1.0=DDPM-류). 0이면 결정적(DDIM).

        # denoiser: dual-branch UNet
        self.denoise_fn = DualBranchUNet(
            down_kind=down_kind, base=base, use_attn_hi=use_attn_hi)

        # ----- BBDM 스케줄 버퍼 -----
        # m_t = t/T 선형. 인덱스 0..T 로 T+1개 만들어 t와 t-1을 쉽게 참조.
        T = num_timesteps
        m_t = np.linspace(0.0, 1.0, T + 1, dtype=np.float64)  # [T+1]
        m_t[0] = 1e-4; m_t[-1] = 1.0 - 1e-4   # 양끝 0/1 회피 (var_t=0 나눗셈 방지)
        # 양 끝 수치안정: t=0 근처는 x0, t=T 근처는 y1 이 되도록 그대로 둬도 OK.
        var_t = 2.0 * self.max_var * (m_t - m_t ** 2)          # δ_t

        self.register_buffer('m_t', torch.tensor(m_t, dtype=torch.float32))
        self.register_buffer('variance_t', torch.tensor(var_t, dtype=torch.float32))

    # 학습 대상 파라미터(UNet 전체). encoder 같은 frozen 모듈이 없으므로 전부.
    def get_parameters(self):
        return self.denoise_fn.parameters()

    # ------------------------------------------------------------------
    # 학습: forward → loss
    #   x0 = PET, y1 = 전체 MRI, y2 = 병변(MRI⊙mask)
    # ------------------------------------------------------------------
    def forward(self, x0, y1, y2, noise=None):
        b = x0.shape[0]
        dev = x0.device
        # timestep 랜덤 샘플 (배치별로 서로 다른 t)
        t = torch.randint(0, self.num_timesteps, (b,), device=dev).long()
        if noise is None:
            noise = torch.randn_like(x0)

        # m_t, var_t 를 배치 t에 맞춰 뽑고 [B,1,1,1,1]로 broadcast
        m = self.m_t[t].view(b, 1, 1, 1, 1)
        var = self.variance_t[t].view(b, 1, 1, 1, 1)
        sigma = torch.sqrt(var.clamp(min=0))

        # bridge 상태 x_t = (1-m)x0 + m·y1 + sigma·ε
        x_t = (1.0 - m) * x0 + m * y1 + sigma * noise
        # 네트워크가 맞출 objective(grad) = m·(y1-x0) + sigma·ε
        target = m * (y1 - x0) + sigma * noise

        # denoiser: 입력은 x_t(bridge)와 y2(병변) 두 개
        pred = self.denoise_fn(x_t, y2, timesteps=t)

        loss = (target - pred).abs().mean()   # L1
        return loss, {'loss': loss.item()}

    # ------------------------------------------------------------------
    # 샘플링(역과정): x_T = y1(전체 MRI)에서 출발 → x_0(PET) 복원
    #   계수는 project_context section 10의 검증된 식과 동일 구조.
    #   objective='grad' 에서 x0 ≈ x_t − pred (한 스텝 x0 추정) 사용.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, y1, y2, clip_denoised=False):
        """section 10 검증 계수. y1=MRI(x_T), y2=병변."""
        b=y1.shape[0]; dev=y1.device
        steps=list(reversed(np.linspace(0,self.num_timesteps,self.sample_step+1,dtype=int).tolist()))
        z_t=y1.clone()
        z_cond=y1  # bridge 종점 (원본 z_sdf 자리)
        for i in range(len(steps)-1):
            cs,ns=steps[i],steps[i+1]
            t=torch.full((b,),cs,device=dev,dtype=torch.long)
            obj=self.denoise_fn(z_t, y2, timesteps=t)
            z0r=z_t - obj
            if clip_denoised: z0r=z0r.clamp(-1,1)
            if ns==0:
                z_t=z0r
            else:
                mt=self.m_t[cs]; mnt=self.m_t[ns]
                vt=self.variance_t[cs]; vnt=self.variance_t[ns]
                s2=torch.clamp((vt - vnt*(1.-mt)**2/((1.-mnt)**2+1e-8))*vnt/(vt+1e-8),min=0)
                z_t=((1.-mnt)*z0r + mnt*z_cond
                     + torch.sqrt(torch.clamp((vnt-s2)/(vt+1e-8),min=0))*(z_t-(1.-mt)*z0r-mt*z_cond)
                     + torch.sqrt(s2)*self.eta*torch.randn_like(z_t))
        return z_t
