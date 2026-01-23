# =============================================
# File: /workspace/stereocraft/scripts/test.py
# ---------------------------------------------
# 目的: Mamba2の簡易動作確認
# =============================================

if __name__ == "__main__":
    from mamba_ssm import Mamba2
    import torch
    mamba = Mamba2(
        d_model=320,
        d_state=256,
        d_conv=4,
        expand=4,
        chunk_size=32,
    ).cuda()
    
    # Execute small tensor first
    x = torch.randn(30720, 8, 320).cuda()
    x = mamba(x)
    print(x.shape)  # (30720, 8, 320) - Works
