
import torch.nn as nn
import segmentation_models_pytorch as smp

class Network(nn.Module):

  def __init__(self,embedding_size=128):
    super(Network,self).__init__()
    self.seg_model = smp.DeepLabV3Plus('resnet101',classes=21,in_channels=3,encoder_weights='imagenet',activation=None)
    self.encoder= self.seg_model.encoder
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Sequential(
                nn.Linear(2048,2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048,embedding_size),
                nn.BatchNorm1d(128)
        ) #2048 for ResNet50 and 101;
    self.contrast=False

  def forward(self,x):
    if self.contrast is True:
        # print('yo')
        x =self.encoder(x)
        x=x[-1] #Taking the last feature map only
        x =self.avgpool(x)
        x = torch.flatten(x, 1)
        x =self.fc(x)
        return x
    else:
        return self.seg_model(x)
