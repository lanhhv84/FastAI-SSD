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
        self.n_classes = n_classes

    @staticmethod
    def create_conv(*arg, **kwargs):
        return nn.Sequential(
            nn.Conv2d(*arg, **kwargs), 
            nn.BatchNorm2d(arg[1]), 
            nn.ReLU()
        )

    def forward(self, data):
        fm = data
        conf = []
        loc = []
        for layer, layer_conf, layer_loc in zip(self.fms, self.fms_conf, self.fms_loc):
            fm = layer(fm)
            conf.append(layer_conf(fm))
            loc.append(layer_loc(fm))
        return loc, conf
    
    def test(self, inp):
        bs = inp.shape[0]

        loc_shape = [[bs, 4*self.fm1_priors, 38, 38], \
            [bs, 4*self.fm2_priors, 19, 19], \
                [4, 4*self.fm3_priors, 10, 10], \
                    [4, 4*self.fm4_priors, 5, 5], \
                        [4, 4*self.fm5_priors, 3, 3], \
                            [4, 4*self.fm6_priors, 1, 1]
        ]
        conf_shape = [[bs, self.n_classes*self.fm1_priors, 38, 38], \
            [bs, self.n_classes*self.fm2_priors, 19, 19], \
                [bs, self.n_classes*self.fm3_priors, 10, 10], \
                    [bs, self.n_classes*self.fm4_priors, 5, 5], \
                        [bs, self.n_classes*self.fm5_priors, 3, 3], \
                            [bs, self.n_classes*self.fm6_priors, 1, 1]]
        loc, conf = self(inp)
        
        for l, shape in zip(loc, loc_shape):
            assert list(l.shape) == shape, "Expect {} Got {}".format(list(l.shape), shape)
        for c, shape in zip(conf, conf_shape):
            assert list(c.shape) == shape, "Expect {} Got {}".format(list(c.shape), shape)


class VGGSSD(SSD):

    def __init__(self, n_classes=2):
        super(VGGSSD, self).__init__(n_classes=n_classes)
        #
        self.fm1 = nn.Sequential(
            SSD.create_conv(3, 64, 3, stride=1, padding=1),
            SSD.create_conv(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),

            SSD.create_conv(64, 128, 3, stride=1, padding=1),
            SSD.create_conv(128, 128, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),

            SSD.create_conv(128, 256, 3, stride=1, padding=1),
            SSD.create_conv(256, 256, 3, stride=1, padding=1),
            SSD.create_conv(256, 256, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            SSD.create_conv(256, 512, 3, stride=1, padding=1),
            SSD.create_conv(512, 512, 3, stride=1, padding=1),
            SSD.create_conv(512, 512, 3, stride=1, padding=1)
        )
        ## (512, 38, 38)
        self.fm1_conf = nn.Conv2d(512, n_classes*self.fm1_priors, 3, padding=1)
        self.fm1_loc = nn.Conv2d(512, 4*self.fm1_priors, 3, padding=1)
        

        self.fm2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            SSD.create_conv(512, 512, 3, stride=1, padding=1),
            SSD.create_conv(512, 512, 3, stride=1, padding=1),
            SSD.create_conv(512, 512, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=1, padding=1),
            SSD.create_conv(512, 1024, 3, stride=1, padding=6, dilation=6),
            SSD.create_conv(1024, 1024, 1, stride=1, padding=0)
        )
        self.fm2_conf = nn.Conv2d(1024, n_classes*self.fm2_priors, 3, padding=1)
        self.fm2_loc = nn.Conv2d(1024, 4*self.fm2_priors, 3, padding=1)
        ## (1024, 19, 19)

        self.fm3 = nn.Sequential(
            SSD.create_conv(1024, 256, 1, padding=0),
            SSD.create_conv(256, 512, 3, stride=2, padding=1),
        )
        self.fm3_conf = nn.Conv2d(512, n_classes*self.fm3_priors, 3, padding=1)
        self.fm3_loc = nn.Conv2d(512, 4*self.fm3_priors, 3, padding=1)
        ## (512, 10, 10)

        self.fm4 = nn.Sequential(
            SSD.create_conv(512, 128, 1, padding=0),
            SSD.create_conv(128, 256, 3, stride=2, padding=1)
        )
        self.fm4_conf = nn.Conv2d(256, n_classes*self.fm4_priors, 3, padding=1)
        self.fm4_loc = nn.Conv2d(256, 4*self.fm4_priors, 3, padding=1)
        ## (256, 5, 5)

        self.fm5 = nn.Sequential(
            SSD.create_conv(256, 128, 1, padding=0),
            SSD.create_conv(128, 256, 3, padding=0)
        )
        self.fm5_conf = nn.Conv2d(256, n_classes*self.fm5_priors, 3, padding=1)
        self.fm5_loc = nn.Conv2d(256, 4*self.fm5_priors, 3, padding=1)
        ## (256, 3, 3)

        self.fm6 = nn.Sequential(
            SSD.create_conv(256, 128, 1, padding=0),
            SSD.create_conv(128, 256, 3, padding=0)
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




class MobileNetSSD(SSD):

    def __init__(self, n_classes=2):
        super(MobileNetSSD, self).__init__(n_classes=n_classes)

        self.fm1 = nn.Sequential(
            MobileNetSSD.conv_dw(3, 64, 1, 1),
            MobileNetSSD.conv_dw(64, 64, 1, 1),
            nn.MaxPool2d(2, stride=2),

            MobileNetSSD.conv_dw(64, 128, 1, 1),
            MobileNetSSD.conv_dw(128, 128, 1, 1),
            nn.MaxPool2d(2, stride=2),

            MobileNetSSD.conv_dw(128, 256, 1, 1),
            MobileNetSSD.conv_dw(256, 256, 1, 1),
            MobileNetSSD.conv_dw(256, 256, 1, 1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            MobileNetSSD.conv_dw(256, 512, 1, 1),
            MobileNetSSD.conv_dw(512, 512, 1, 1),
            MobileNetSSD.conv_dw(512, 512, 1, 1)
        )
        ## (512, 38, 38)
        self.fm1_conf = nn.Conv2d(512, n_classes*self.fm1_priors, 3, padding=1)
        self.fm1_loc = nn.Conv2d(512, 4*self.fm1_priors, 3, padding=1)
        

        self.fm2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            MobileNetSSD.conv_dw(512, 512, 1, 1),
            MobileNetSSD.conv_dw(512, 512, 1, 1),
            MobileNetSSD.conv_dw(512, 512, 1, 1),
            nn.MaxPool2d(3, stride=1, padding=1),
            MobileNetSSD.conv_dw(512, 1024, 1, 6, 6),
            MobileNetSSD.create_conv(1024, 1024, 1, stride=1, padding=0)
        )
        self.fm2_conf = nn.Conv2d(1024, n_classes*self.fm2_priors, 3, padding=1)
        self.fm2_loc = nn.Conv2d(1024, 4*self.fm2_priors, 3, padding=1)
        ## (1024, 19, 19)

        self.fm3 = nn.Sequential(
            MobileNetSSD.create_conv(1024, 256, 1, padding=0),
            MobileNetSSD.conv_dw(256, 512, 2, 1),
        )
        self.fm3_conf = nn.Conv2d(512, n_classes*self.fm3_priors, 3, padding=1)
        self.fm3_loc = nn.Conv2d(512, 4*self.fm3_priors, 3, padding=1)
        ## (512, 10, 10)

        self.fm4 = nn.Sequential(
            MobileNetSSD.create_conv(512, 128, 1, padding=0),
            MobileNetSSD.conv_dw(128, 256, 2, 1)
        )
        self.fm4_conf = nn.Conv2d(256, n_classes*self.fm4_priors, 3, padding=1)
        self.fm4_loc = nn.Conv2d(256, 4*self.fm4_priors, 3, padding=1)
        ## (256, 5, 5)

        self.fm5 = nn.Sequential(
            MobileNetSSD.create_conv(256, 128, 1, padding=0),
            MobileNetSSD.conv_dw(128, 256, 1, 0)
        )
        self.fm5_conf = nn.Conv2d(256, n_classes*self.fm5_priors, 3, padding=1)
        self.fm5_loc = nn.Conv2d(256, 4*self.fm5_priors, 3, padding=1)
        ## (256, 3, 3)

        self.fm6 = nn.Sequential(
            MobileNetSSD.create_conv(256, 128, 1, padding=0),
            MobileNetSSD.conv_dw(128, 256, 1, 0)
        )
        self.fm6_conf = nn.Conv2d(256, n_classes*self.fm6_priors, 3, padding=1)
        self.fm6_loc = nn.Conv2d(256, 4*self.fm6_priors, 3, padding=1)
        ## (256, 1, 1)

        self.fms = [self.fm1, self.fm2, self.fm3, self.fm4, self.fm5, self.fm6]
        self.fms_conf = [self.fm1_conf, self.fm2_conf, self.fm3_conf, self.fm4_conf, self.fm5_conf, self.fm6_conf]
        self.fms_loc = [self.fm1_loc, self.fm2_loc, self.fm3_loc, self.fm4_loc, self.fm5_loc, self.fm6_loc]

    @staticmethod
    def conv_dw(inp, oup, stride, pad, dil=1):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, pad, dilation=dil, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    