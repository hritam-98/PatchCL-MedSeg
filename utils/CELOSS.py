import segmentation_models_pytorch as smp
import torch


class CE_loss(smp.utils.losses.CrossEntropyLoss):
    def __init__(self):
        super(CE_loss,self).__init__()
        #self.weights=torch.ones(21)
        #self.weights[0]=0.05
        self.ce_loss=smp.utils.losses.CrossEntropyLoss()
        self.matrix_mult=torch.ones(21,512,512)*torch.arange(21).view(21,1,1)

    def forward(self,prediction,target):
        self.matrix_mult=(self.matrix_mult).to(dev)
        target = target*self.matrix_mult.unsqueeze(0)
        target = (torch.sum(target,dim=1)).type(torch.long)
        return self.ce_loss(prediction,target)