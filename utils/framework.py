import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from tqdm import tqdm_notebook as tqdm
import cv2
#from sklearn import metrics
#from sklearn.metrics import jaccard_score as js
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import auc as auc
#from PIL import Image, ImageOps
#from torch.autograd import Variable as v
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
%matplotlib inline
#import pandas as pd
import os
from google.colab.patches import cv2_imshow
import pickle
import nibabel as nib
import pathlib
from skimage import transform
from sklearn.manifold import TSNE

root_path = '/content/drive/MyDrive/Data/ACDC/training_ACDC'
input_size = (3,256,256) #for kaggle 448
batch_size = 1
learning_rate = 0.000001
epochs = 500

INITAL_EPOCH_LOSS = 10000
NUM_EARLY_STOP = 20
NUM_UPDATE_LR = 5

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def loader(img_path, mask_path, noise=0):
    img = nib.load(img_path).get_data()
    mask = nib.load(mask_path).get_data()
    #print(img.shape, mask.shape)
    img, mask = np.array(img), np.array(mask)
    I, M = [], []
    for i in range(0,4):
        im, ms = img[:,:,i][30:186,20:196], mask[:,:,i][30:186,20:196]
        im, ms = cv2.resize(im, (128,128),cv2.INTER_AREA), cv2.resize(ms, (128,128),cv2.INTER_AREA)
        I.append(im)
        M.append(ms)
    img = np.array(I, dtype=float)/255.0
    mask = np.array(M)
    img = img + noise*np.random.rand(img.shape[0], img.shape[1], img.shape[2])
    #mask = mask[np.newaxis,:,:]
    
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    
    return img, mask

def read_dataset(root_path):
    images = []
    labels = []

    for image_name in sorted(os.listdir(root_path)):
        image_path = os.path.join(root_path, image_name) #.split('.')[0] + '.jpg')
        if os.path.isfile(image_path+'/'+image_name+'_frame01.nii.gz'):
            images.append(image_path+'/'+image_name+'_frame01.nii.gz')
            labels.append(image_path+'/'+image_name+'_frame01_gt.nii.gz')
        else:
            images.append(image_path+'/'+image_name+'_frame04.nii.gz')
            labels.append(image_path+'/'+image_name+'_frame04_gt.nii.gz')
        #images.append(image_path+'/'+image_name+'_frame01.nii.gz')
        #labels.append(image_path+'/'+image_name+'_frame01_gt.nii.gz')
    return images, labels

class Dataset(Dataset):

    def __init__(self, root_path, noise=0):
        self.root = root_path
        self.images, self.labels = read_dataset(self.root)
        self.noise = noise
        print('Num Images:', len(self.images), 'Num Labes:', len(self.labels))
        print('images: ', self.images)
        print('labels: ', self.labels)

    def __getitem__(self, index):
        #print(self.images[index], self.labels[index])
        img, mask = loader(self.images[index], self.labels[index], self.noise)
        img = torch.tensor(img, dtype = torch.float32)
        mask = torch.tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

#U-Net Accessories
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        
        
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)
        

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class BatchNorm(nn.Module):
  def init(self, out_channels):
    super(BatchNorm, self).init()
    #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    #x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class UNet(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.act = nn.Sigmoid()

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up4 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128, bilinear) 
        self.up1 = Up(128, 64, bilinear)

        self.inc = DoubleConv(3, 64)
        self.outconv = OutConv(64, 3)

    def forward(self, x, factor=0.001, visual = False):


        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        xb = self.down4(x4)

        y4 = self.up4(xb, x4)
        y3 = self.up3(y4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        
        out = self.outconv(y1)
        out = self.act(out)
        return out

class Encoder_normie(nn.Module):
    def __init__(self, bilinear=False):
        super(Encoder_normie, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.inc = DoubleConv(1, 64)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.down4(x4)
        return xb, x4, x3, x2, x1

class Block(nn.Module):
    def __init__(self, in_channels=64):
        super(Block, self).__init__()
        self.conv1 = BasicConv2d(in_channels, in_channels, 3,1,1)
        self.conv2 = BasicConv2d(in_channels, in_channels, 3,1,1)
    
    def forward(self, x):
        xn = self.conv1(x)
        xn = self.conv2(xn)
        x = x + xn
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        l1 = []
        l2 = []
        l3 = []
        l4 = []
        self.inc = BasicConv2d(1, 64, 3, 1, 1)
        self.out = BasicConv2d(1024, 1024, 1)
        self.mp = nn.MaxPool2d(2)
        for i in range(3):
            l1.append(Block(64))
        self.lm1 = nn.Conv2d(64,128,1)
        for i in range(4):
            l2.append(Block(128))
        self.lm2 = nn.Conv2d(128,256,1)
        for i in range(6):
            l3.append(Block(256))
        self.lm3 = nn.Conv2d(256,512,1)
        for i in range(3):
            l4.append(Block(512))
        self.lm4 = nn.Conv2d(512,1024,1)
        self.layer1 = nn.Sequential(*l1)
        self.layer2 = nn.Sequential(*l2)
        self.layer3 = nn.Sequential(*l3)
        self.layer4 = nn.Sequential(*l4)
    
    def forward(self, x):
        x = self.inc(x)
        x = self.layer1(x)
        x = self.lm1(x)
        x = self.mp(x)

        x = self.layer2(x)
        x = self.lm2(x)
        x = self.mp(x)
        
        x = self.layer3(x)
        x = self.lm3(x)
        x = self.mp(x)
        
        x = self.layer4(x)
        x = self.lm4(x)
        x = self.mp(x)
        
        x = self.out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, bilinear=False):
        super(Decoder, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up4 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128, bilinear) 
        self.up1 = Up(128, 64, bilinear)
        self.act = nn.Sigmoid()
        self.outconv = OutConv(64, 1)
    
    def forward(self, xb, x4, x3, x2, x1):
        y4 = self.up4(xb, x4)
        y3 = self.up3(y4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        
        out = self.outconv(y1)
        out = self.act(out)
        return out

class Grep(nn.Module):
    def __init__(self):
        super(Grep, self).__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(65536, 3200)
        self.ln1 = nn.LayerNorm(3200)
        self.l2 = nn.Linear(3200, 128)
        self.ln2 = nn.LayerNorm(128)
    def forward(self, x):
        x = self.flat(x)
        x = self.ln1(self.l1(x))
        x = self.ln2(self.l2(x))
        return F.softmax(x, dim=1)

def convert(A, Ttnp = True):
    if Ttnp == True: return A.detach().cpu().numpy()
    else: return torch.tensor(A).to(device)

class Visualize():
    def __init__(self):
        super(Visualize, self).__init__()
        self.lst = []
    
    def register(self, x):
        self.lst.append(x)
    
    def make_numpy(self, vector_length=128):
        self.dataset = np.array(self.lst, dtype=float)
        self.shape = self.dataset.shape
        self.dimension = vector_length
    
    def make_representation(self):
        emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(self.dataset)
        self.embeddings = emb
    
    def plot(self):
        plt.scatter(self.embeddings[:,0], self.embeddings[:,1], s=5)
        plt.show()

class contrastive_loss(nn.Module):
    def __init__(self):
        super(contrastive_loss, self).__init__()

    def mag(self, x):
        x = x**2
        s = x.sum()
        s = s**(1/2)
        return s
    
    def cosine_similarity(self, x, y):
        S = (x*y).sum()
        S = S/(self.mag(x) * self.mag(y))
        return S
    
    def forward(self, pos, neg, t=1):
        N, D = torch.zeros([1]), torch.zeros([1])#.to(device)
        p = len(pos)
        for i in range(1,p):
            cos = self.cosine_similarity(pos[i], pos[0]).cpu()
            N += torch.exp(cos/t)
        #print(self.N)
        n = len(neg)
        for i in range(n):
            cos = self.cosine_similarity(pos[0], neg[i]).cpu()
            D += torch.exp(cos/t)
        #print(self.D)
        loss = - torch.log(N/(N+D))
        #N, D = convert(N), convert(D)
        return loss


def perturb(x):
    x = convert(x)[0,0,10:118,10:118]
    x = cv2.resize(x, (128,128),cv2.INTER_AREA)
    x = convert(x, False)
    x = x.unsqueeze(0).unsqueeze(0)
    return x

def adversarial(x, xp, Enc, G, loss):
    x_att = x
    x_att.requires_grad_()
    x_a = G(Enc.forward(x_att))
    adg = loss.cosine_similarity(x_a,xp)
    adg.backward(retain_graph=True)
    sal, _ = torch.max(x_att.grad.data.abs(), dim=1)
    sal = (sal - sal.min())/sal.max()
    # Visual Inspection
    #print('Similarity: ',adg)
    
    noise = torch.rand(x.shape).to(device)
    #print('Saliency Map and Noise Map: ')
    #cv2_imshow(sal[0,:,:].detach().cpu().numpy()*255.0)
    #cv2_imshow(noise[0,0,:,:].detach().cpu().numpy()*255.0)
    noise = noise*sal*0.7
    xa = (x + noise)
    #print('Adversarial Image: ')
    #cv2_imshow(xa[0,0,:,:].detach().cpu().numpy()*255.0)
    adg = 0
    return xa


class MyFrame():
    def __init__(self, encoder, learning_rate, device, evalmode=False):
        self.Enc = encoder().to(device)
        self.G = Grep().to(device)
        self.optimizer = torch.optim.Adam(params=list(self.Enc.parameters()) + list(self.G.parameters()), lr=learning_rate, weight_decay=0.0001)
        self.loss = contrastive_loss().to(device)
        self.lr = learning_rate

    
    def set_input(self, img_batch, mask_batch=None):
        self.img = img_batch
        self.mask = mask_batch
        
    def optimize(self):
        self.optimizer.zero_grad()
        c = self.img.shape[1]
        b = self.img.shape[0]
        L = 0
        preds = []
        for i in range(c):
            pos = []
            neg = []
            im1 = self.img
            x = im1[0,i,:,:]
            im1 = im1[0,i,:,:]
            
            x = x.unsqueeze(0).unsqueeze(0)
            #print('Image: ')
            #cv2_imshow(x[0,0,:,:].detach().cpu().numpy()*255.0)
            xp = perturb(x) #Perturbation
            #print('Perturbed Image: ')
            #cv2_imshow(xp[0,0,:,:].detach().cpu().numpy()*255.0)
            xo = self.G(self.Enc.forward(x))
            xp = self.G(self.Enc.forward(xp))
            #xa = adversarial(x, xp, self.Enc, self.G, self.loss)
            #xa = self.G(self.Enc.forward(xa))
            pos = [xo, xp]
            preds.append(xo)
            
            for j in range(c):
                if j==i: continue
                n = self.img[0,j,:,:]
                n = n.unsqueeze(0).unsqueeze(0)
                #print('Image: ')
                #cv2_imshow(n[0,0,:,:].detach().cpu().numpy()*255.0)
                no = self.G(self.Enc.forward(n))
                neg.append(no)
                np = perturb(n)
                #print('Perturbed Image: ')
                #cv2_imshow(np[0,0,:,:].detach().cpu().numpy()*255.0)
                np = self.G(self.Enc.forward(np))
                neg.append(np)
                
                #na = adversarial(n, np, self.Enc, self.G, self.loss)
                #na = self.G(self.Enc.forward(na))
                #neg.append(na)
                

            L += self.loss.forward(pos,neg,0.1)
        L.backward()
        self.optimizer.step()
        return L.item(), preds
        
    def save(self, path):
        #torch.save(self.Enc.state_dict(), path + '/' + 'pre_enc_resnet34_cropped' + '.pth')
        #torch.save(self.G.state_dict(), path + '/' + 'pre_G_resnet34_cropped' + '.pth')
        torch.save(self.Enc.state_dict(), path + '/' + 'pre_enc_resnet34_cropped_KC' + '.pth')
        torch.save(self.G.state_dict(), path + '/' + 'pre_G_resnet34_cropped_KC' + '.pth')

    def load(self, path):
        #self.Enc.load_state_dict(torch.load(path + '/' + 'pre_enc_resnet34_cropped' + '.pth'))
        #self.G.load_state_dict(torch.load(path + '/' + 'pre_G_resnet34_cropped' + '.pth'))
        self.Enc.load_state_dict(torch.load(path + '/' + 'pre_enc_resnet34_cropped_KC' + '.pth'))
        self.G.load_state_dict(torch.load(path + '/' + 'pre_G_resnet34_cropped_KC' + '.pth'))

    def update_lr(self, new_lr, factor=False):

        if factor:
            new_lr = self.lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print ('update learning rate: %f -> %f' % (self.lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.lr, new_lr))
        self.lr = new_lr

class proposed_loss(nn.Module):
    def __init__(self, batch=True):
        super(proposed_loss, self).__init__()
        self.batch = batch
        self.mae_loss = torch.nn.L1Loss()
        self.bce_loss = torch.nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def iou_loss(self, inputs, targets):
        smooth = 0.0
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)
        
        intersection = (inputs * targets).sum(1).sum(1).sum(1)
        total = (inputs + targets).sum(1).sum(1).sum(1)
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return (1 - IoU.mean())

    def forward(self, y_true, y_pred):
        a = self.mae_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        c = self.bce_loss(y_pred, y_true)
        d = self.iou_loss(y_pred, y_true)
        loss = 0.15*a + 0.4*b  + 0.15*c + 0.3*d
        return loss


