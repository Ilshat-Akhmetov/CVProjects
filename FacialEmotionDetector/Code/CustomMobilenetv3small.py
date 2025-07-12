import torch
import torchvision.models as models


class CustomMobilenetv3small(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.model.classifier[3]=torch.nn.Linear(1024, out_dim)
        weights = self.model.features[0][0].weight.mean(dim=1, keepdim=True)
        with torch.no_grad():
            self.model.features[0][0]=torch.nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # get gray images insted rgb
            self.model.features[0][0].weight.copy_(weights)
        for param in self.model.parameters():
            param.requires_grad = True
        n_layers_to_freeze = 0
        for i in range(n_layers_to_freeze):
            for param in self.model.features[i].parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: torch.Tensor with shape (batch_size, 1, img_size, img_size)
        output: torch.Tensor with shape (batch_size, n_classes)
        """
        outp = self.model(x)
        return outp