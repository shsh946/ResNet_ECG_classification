import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=7, padding=3, stride=stride),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(outchannel, outchannel, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(outchannel)
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(outchannel),
            nn.MaxPool1d(stride)
        )

    def forward(self, x):
        out = self.left(x)
        res = self.shortcut(x)
        out += res
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=7):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.layer1 = self.make_layer(ResidualBlock, inchannels=64, channels=64, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, inchannels=64, channels=128, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, inchannels=128, channels=128, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, inchannels=128, channels=256, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, inchannels=256, channels=256, stride=2)
        self.layer6 = self.make_layer(ResidualBlock, inchannels=256, channels=512, stride=2)
        self.layer7 = self.make_layer(ResidualBlock, inchannels=512, channels=512, stride=2)
        self.layer8 = self.make_layer(ResidualBlock, inchannels=512, channels=1024, stride=2)

        self.bn = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(1024*4, num_classes)


    def make_layer(self, block, inchannels, channels, stride):
        strides = [stride] + [1]
        input_c, output_c = inchannels, channels
        layers = []
        for stride in strides:
            layers.append(block(inchannel=input_c, outchannel=output_c, stride=stride))
            input_c = output_c
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x.unsqueeze(1))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def ResNet18():
    return ResNet(ResidualBlock)


