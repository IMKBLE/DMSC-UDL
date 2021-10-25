import torch.nn as nn
import torch


class Networks(nn.Module):
    def __init__(self):
        super(Networks, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.model1 = nn.Linear(1470, 10)
        self.model2 = nn.Linear(1470, 10)
        self.weight = nn.Parameter(1.0e-4 * torch.ones(2000, 2000))

    def forward(self, input1, input2):
        output1 = self.encoder1(input1)
        output1 = self.decoder1(output1)

        output2 = self.encoder2(input2)
        output2 = self.decoder2(output2)

        return output1, output2

    def forward2(self, input1, input2):
        coef = self.weight - torch.diag(torch.diag(self.weight))

        z1 = self.encoder1(input1)
        z1 = z1.view(2000, 1470)
        z11 = self.model1(z1)
        zcoef1 = torch.matmul(coef, z1)
        output1 = zcoef1.view(2000, 30, 7, 7)
        output1 = self.decoder1(output1)

        z2 = self.encoder2(input2)
        z2 = z2.view(2000, 1470)
        z22 = self.model2(z2)
        zcoef2 = torch.matmul(coef, z2)
        output2 = zcoef2.view(2000, 30, 7, 7)
        output2 = self.decoder2(output2)

        return z11, z22, z1, zcoef1, output1, coef, z2, zcoef2, output2






