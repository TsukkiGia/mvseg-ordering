import torch

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Dice score between two tensors.
    
    Args:
        pred: torch.Tensor of shape (N, ...) — predicted mask (binary or probabilities)
        target: torch.Tensor of shape (N, ...) — ground-truth mask (binary)
        eps: float — small constant to avoid division by zero
    
    Returns:
        torch.Tensor (scalar): Dice score
    """
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    
    return (2. * intersection + eps) / (union + eps)
