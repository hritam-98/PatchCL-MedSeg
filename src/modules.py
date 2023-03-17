import torch.nn as nn


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, x):
        return self.f(x)


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        residual = self.block(x)
        return x + residual


class ConfidenceNetwork(nn.Module):
    conv2d_settings = [
        dict(kernel_size=3, stride=1, padding=pd, dilation=pd, bias=False)
        for pd in (1, 2, 3, 5, 10, 20)
    ]

    def __init__(self, in_channels: int = 1, internal_channels: int = 8, out_channels: int = 1):
        super().__init__()
        first_conv_block = Conv2dBlock(
            in_channels, internal_channels, **self.conv2d_settings[0]
        )
        residual_blocks = [
            ResidualBlock(
                nn.Sequential(
                    Conv2dBlock(internal_channels, internal_channels, **kw),
                    Conv2dBlock(internal_channels, internal_channels, **kw),
                )
            )
            for kw in self.conv2d_settings
        ]
        last_conv_block = nn.Conv2d(
            internal_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        self.f = nn.Sequential(
            first_conv_block, *residual_blocks, last_conv_block, nn.Sigmoid()
        )

    def forward(self, x):
        return self.f(x)


class FeatureNetwork(nn.Module):
    def __init__(self, in_feature_channels: int = 1):
        super().__init__()
        self.f = nn.Sequential(
            Conv2dBlock(in_feature_channels, 8, kernel_size=3),
            Conv2dBlock(8, 8, kernel_size=3, stride=2, dilation=2, bias=False),
            Conv2dBlock(8, 1, kernel_size=3, stride=2, dilation=2, bias=False),
            nn.Flatten(),
            nn.Linear(81, 32),
        )

    def forward(self, x):
        return self.f(x)
