import torch
import torchvision.models as models


class CustomResnet18(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        weights = self.model.conv1.weight.mean(dim=1, keepdim=True)
        with torch.no_grad():
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # get gray images instead rgb
            self.model.conv1.weight.copy_(weights)
        # for param in self.model.parameters():
        #     param.requires_grad = True

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, out_dim)
        
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        for param in self.model.layer3.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: torch.Tensor with shape (batch_size, 1, img_size, img_size)
        output: torch.Tensor with shape (batch_size, n_classes)
        """
        outp = self.model(x)
        return outp
        