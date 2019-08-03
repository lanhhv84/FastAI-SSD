import torch
import torch.nn as nn

class SSD(nn.Module):

    def __init__(self, n_classes=2):
        super(SSD, self).__init__()
        self.fm1_priors = 4
        self.fm2_priors = 6
        self.fm3_priors = 6
        self.fm4_priors = 6
        self.fm5_priors = 4
        self.fm6_priors = 4

        #
        self.fm1 = nn.Sequential(
            *SSD.create_conv(3, 64, 3, stride=1, padding=1),
            *SSD.create_conv(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),

            *SSD.create_conv(64, 128, 3, stride=1, padding=1),
            *SSD.create_conv(128, 128, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),

            *SSD.create_conv(128, 256, 3, stride=1, padding=1),
            *SSD.create_conv(256, 256, 3, stride=1, padding=1),
            *SSD.create_conv(256, 256, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            *SSD.create_conv(256, 512, 3, stride=1, padding=1),
            *SSD.create_conv(512, 512, 3, stride=1, padding=1),
            *SSD.create_conv(512, 512, 3, stride=1, padding=1)
        )

        self.fm1_conf = nn.Conv2d(512, n_classes*self.fm1_priors, 3, padding=1)
        self.fm1_loc = nn.Conv2d(512, 4*self.fm1_priors, 3, padding=1)
        ## (512, 38, 38)

        self.fm2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            *SSD.create_conv(512, 512, 3, stride=1, padding=1),
            *SSD.create_conv(512, 512, 3, stride=1, padding=1),
            *SSD.create_conv(512, 512, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=1, padding=1),
            *SSD.create_conv(512, 1024, 3, stride=1, padding=6, dilation=6),
            *SSD.create_conv(1024, 1024, 1, stride=1, padding=0)
        )
        self.fm2_conf = nn.Conv2d(1024, n_classes*self.fm2_priors, 3, padding=1)
        self.fm2_loc = nn.Conv2d(1024, 4*self.fm2_priors, 3, padding=1)
        ## (1024, 19, 19)

        self.fm3 = nn.Sequential(
            *SSD.create_conv(1024, 256, 1, padding=0),
            *SSD.create_conv(256, 512, 3, stride=2, padding=1),
        )
        self.fm3_conf = nn.Conv2d(512, n_classes*self.fm3_priors, 3, padding=1)
        self.fm3_loc = nn.Conv2d(512, 4*self.fm3_priors, 3, padding=1)
        ## (512, 10, 10)

        self.fm4 = nn.Sequential(
            *SSD.create_conv(512, 128, 1, padding=0),
            *SSD.create_conv(128, 256, 3, stride=2, padding=1)
        )
        self.fm4_conf = nn.Conv2d(256, n_classes*self.fm4_priors, 3, padding=1)
        self.fm4_loc = nn.Conv2d(256, 4*self.fm4_priors, 3, padding=1)
        ## (256, 5, 5)

        self.fm5 = nn.Sequential(
            *SSD.create_conv(256, 128, 1, padding=0),
            *SSD.create_conv(128, 256, 3, padding=0)
        )
        self.fm5_conf = nn.Conv2d(256, n_classes*self.fm5_priors, 3, padding=1)
        self.fm5_loc = nn.Conv2d(256, 4*self.fm5_priors, 3, padding=1)
        ## (256, 3, 3)

        self.fm6 = nn.Sequential(
            *SSD.create_conv(256, 128, 1, padding=0),
            *SSD.create_conv(128, 256, 3, padding=0)
        )
        self.fm6_conf = nn.Conv2d(256, n_classes*self.fm6_priors, 3, padding=1)
        self.fm6_loc = nn.Conv2d(256, 4*self.fm6_priors, 3, padding=1)
        ## (256, 1, 1)

        self.fms = [self.fm1, self.fm2, self.fm3, self.fm4, self.fm5, self.fm6]
        self.fms_conf = [self.fm1_conf, self.fm2_conf, self.fm3_conf, self.fm4_conf, self.fm5_conf, self.fm6_conf]
        self.fms_loc = [self.fm1_loc, self.fm2_loc, self.fm3_loc, self.fm4_loc, self.fm5_loc, self.fm6_loc]


    def forward(self, data):
        fm = data
        conf = []
        loc = []
        for layer, layer_conf, layer_loc in zip(self.fms, self.fms_conf, self.fms_loc):
            fm = layer(fm)
            conf.append(layer_conf(fm))
            loc.append(layer_loc(fm))
        return loc, conf

    @staticmethod
    def create_conv(*arg, **kwargs):
        return nn.Conv2d(*arg, **kwargs), nn.BatchNorm2d(arg[1]), nn.ReLU()

