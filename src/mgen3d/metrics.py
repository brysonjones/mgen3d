import torch

def calc_psnr_from_mse(mse:torch.Tensor) -> float:
    # Calculates peak-signal-to-noise ratio from mean-squared-error, in base-10
    return -10.0 * torch.log10(mse)