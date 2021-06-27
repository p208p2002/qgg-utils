import torch
import torch.nn as nn

class NegativeLabelLoss(nn.Module):
    """
    https://www.desmos.com/calculator/9oaqcjayrw
    """
    def __init__(self, ignore_index=-100, reduction='mean',alpha=1.0,beta=0.8):
        super(NegativeLabelLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.alpha = alpha
        self.beta = beta
        self.nll = nn.NLLLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, logits, target):
        nsoftmax = self.softmax(logits)
        nsoftmax = torch.where(
                nsoftmax<=torch.tensor([self.beta],dtype=logits.dtype).to(logits.device),
                torch.tensor([0.0],requires_grad=True,dtype=logits.dtype).to(logits.device),
                nsoftmax
            )
        nsoftmax = torch.clamp((1.0 - nsoftmax), min=1e-32)
        return self.nll(torch.log(nsoftmax) * self.alpha, target)