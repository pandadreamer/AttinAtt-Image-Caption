import torch
import torch.nn as nn
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=7):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        #att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)
        att = x.squeeze().permute(1,2,0)
        #att = nn.MaxPool2d(2, stride = 2)(x).squeeze().permute(1, 2, 0)
        #print(att.shape)
        #print(fc.shape)

        return fc, att

