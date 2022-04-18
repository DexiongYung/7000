import torch
from math import prod
from torchvision import models

class BackBoneNN(torch.nn.Module):
    def __init__(self, input_width: int, input_height: int):
        super(BackBoneNN, self).__init__()
        self.encoder = models.resnet18()
        self.width = input_width
        self.height = input_height
        test_input = torch.zeros(1, 3, input_width, input_height)
        
        with torch.no_grad():
            enc_out = self.encoder(test_input)
            self.out_len = prod(enc_out.shape)
        
    def forward(self, obs: torch.Tensor):
        return torch.flatten(self.encoder(obs), start_dim=1)
