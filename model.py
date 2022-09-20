import sys

sys.path.append("PerceptualSimilarity\\")
import os
import utils
import torch
import numpy as np
from torch import nn
import torchgeometry
from kornia import color
import torch.nn.functional as F
from torchvision import transforms


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class StegaStampEncoder(nn.Module):
    def __init__(self):
        super(StegaStampEncoder, self).__init__()
        self.hidden_dense = Dense(100, 7500, activation='relu', kernel_initializer='he_normal')
        self.embedder = ImageEmbedder()
        self.conv1 = Conv2D(6, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(70, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)
        self.embedder = ImageEmbedder()

    def forward(self, image):
        image = image - .5
        hidden = self.embedder(image)
        hidden = self.hidden_dense(hidden)
        hidden = hidden.reshape(-1, 3, 50, 50)
        hidden = nn.Upsample(scale_factor=(8, 8))(hidden)
        inputs = torch.cat([hidden, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return residual

class GridReduction(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(GridReduction, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(2, 2))
        )

        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        x = torch.cat([o1, o2], dim=1)
        return x
    
class AuxClassifier(nn.Module):
    def __init__(self, in_fts, num_classes):
        super(AuxClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=128, kernel_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(5 * 5 * 128, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        N = x.shape[0]
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        return x

class Inceptionx5(nn.Module):
    def __init__(self, in_fts, out_fts, n=7):
        super(Inceptionx5, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[0], kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(1, n), stride=(1, 1),
                      padding=(0, n // 2)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(n, 1), stride=(1, 1),
                      padding=(n // 2, 0)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(1, n), stride=(1, 1),
                      padding=(0, n // 2)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(n, 1), stride=(1, 1),
                      padding=(n // 2, 0)),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[1], kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=(1, n), stride=(1, 1),
                      padding=(0, n // 2)),
            nn.Conv2d(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=(n, 1), stride=(1, 1),
                      padding=(n // 2, 0)),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[2], kernel_size=(1, 1), stride=(1, 1))
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[3], kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o1, o2, o3, o4], dim=1)
        return x


class Inceptionx2(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Inceptionx2, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[0] // 4, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=out_fts[0] // 4, out_channels=out_fts[0] // 4, kernel_size=(3, 3), stride=(1, 1),
                      padding=1)
        )
        self.subbranch1_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_fts[0] // 4, out_channels=out_fts[0], kernel_size=(1, 3), stride=(1, 1),
                      padding=(0, 3 // 2))
        )
        self.subbranch1_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_fts[0] // 4, out_channels=out_fts[1], kernel_size=(3, 1), stride=(1, 1),
                      padding=(3 // 2, 0))
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[2] // 4, kernel_size=(1, 1))
        )
        self.subbranch2_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_fts[2] // 4, out_channels=out_fts[2], kernel_size=(1, 3), stride=(1, 1),
                      padding=(0, 3 // 2))
        )
        self.subbranch2_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_fts[2] // 4, out_channels=out_fts[3], kernel_size=(3, 1), stride=(1, 1),
                      padding=(3 // 2, 0))
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[4], kernel_size=(1, 1), stride=(1, 1))
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[5], kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o11 = self.subbranch1_1(o1)
        o12 = self.subbranch1_2(o1)
        o2 = self.branch2(input_img)
        o21 = self.subbranch2_1(o2)
        o22 = self.subbranch2_2(o2)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o11, o12, o21, o22, o3, o4], dim=1)
        return x
    
class Inceptionx3(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Inceptionx3, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[0], kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(3, 3), stride=(1, 1), padding=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[1], kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[2], kernel_size=(1, 1), stride=(1, 1))
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts[3], kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o1, o2, o3, o4], dim=1)
        return x
    
class MyInception_v3(nn.Module):
    def __init__(self, in_fts=3, num_classes=2):
        super(MyInception_v3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(num_features=32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=64)
        )
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3, 3), stride=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=80, out_channels=192, kernel_size=(3, 3), stride=(2, 2))
        self.conv6 = nn.Conv2d(in_channels=192, out_channels=288, kernel_size=(3, 3), stride=(1, 1), padding=1)

        list_incept = [Inceptionx3(in_fts=288, out_fts=[96, 96, 96, 96]),
                       Inceptionx3(in_fts=4 * 96, out_fts=[96, 96, 96, 96]),
                       Inceptionx3(in_fts=4 * 96, out_fts=[96, 96, 96, 96])]

        self.inceptx3 = nn.Sequential(*list_incept)
        self.grid_redn_1 = GridReduction(in_fts=4 * 96, out_fts=384)
        self.aux_classifier = AuxClassifier(768, num_classes)

        list_incept = [Inceptionx5(in_fts=768, out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160, out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160, out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160, out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160, out_fts=[160, 160, 160, 160])]

        self.inceptx5 = nn.Sequential(*list_incept)
        self.grid_redn_2 = GridReduction(in_fts=4 * 160, out_fts=640)

        list_incept = [Inceptionx2(in_fts=1280, out_fts=[256, 256, 192, 192, 64, 64]),
                       Inceptionx2(in_fts=1024, out_fts=[384, 384, 384, 384, 256, 256])]

        self.inceptx2 = nn.Sequential(*list_incept)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = Dense(2048, 200, activation='relu')
        self.fc2 = Dense(200, num_classes, activation=None)

    def forward(self, input_img):
        input_img = input_img - 0.5
        N = input_img.shape[0]
        x = self.conv1(input_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.inceptx3(x)
        x = self.grid_redn_1(x)
        aux_out = self.aux_classifier(x)
        x = self.inceptx5(x)
        x = self.grid_redn_2(x)
        x = self.inceptx2(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class StampDetector(nn.Module):
    def __init__(self):
        super(StampDetector, self).__init__()
        self.model = nn.Sequential(
            Conv2D(3, 8, 3, strides=2, activation='relu'),
            Conv2D(8, 16, 3, strides=2, activation='relu'),
            Conv2D(16, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 1, 3, activation='relu'))
        self.fc1 = Dense(25*25, 100, activation='relu')
        self.fc2 = Dense(100, 2, activation=None)
#         self.activate = nn.Sigmoid()
#         self.activate = nn.Softmax()

    def forward(self, image):
        x = image - 0.5 # make pixel value in [-0.5, 0.5]
        x = self.model(x)
        x = torch.flatten(x, 1, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class ImageEmbedder(nn.Module):
    def __init__(self):
        super(ImageEmbedder, self).__init__()
        self.model = nn.Sequential(
            Conv2D(3, 8, 3, strides=2, activation='relu'),
            Conv2D(8, 16, 3, strides=2, activation='relu'),
            Conv2D(16, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 1, 3, activation='relu'))
        self.fc1 = Dense(25*25, 100, activation='relu')

    def forward(self, image):
        x = self.model(image)
        x = torch.flatten(x, 1, -1)
        x = self.fc1(x)
        return x