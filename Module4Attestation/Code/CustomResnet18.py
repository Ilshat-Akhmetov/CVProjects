import torch
from torchvision.models import resnet18
import torchvision.models as models


class CustomResnet18(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.model = resnet18()#weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, out_dim)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: torch.Tensor with shape (batch_size, 1, 96, 96)
        output: torch.Tensor with shape (batch_size, 15, 2)
        """
        y = self.model(x)
        sh = y.shape
        outp = y.reshape(sh[0], sh[1]//2, 2)
        return outp
        