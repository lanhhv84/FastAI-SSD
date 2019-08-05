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
            SEConv2d(3, 64, kernel_size=3, stride=1, padding=1),
            ResBlock([
                SEConv2d(64, 64, kernel_size=3, stride=1, padding=1)
            ]),
            nn.MaxPool2d(2, stride=2),
            SEConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ResBlock([
                SEConv2d(128, 128, kernel_size=3, stride=1, padding=1)
            ]),
            nn.MaxPool2d(2, stride=2),

            SEConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            ResBlock([
                SEConv2d(256, 256, kernel_size=3, stride=1, padding=1),
                SEConv2d(256, 256, kernel_size=3, stride=1, padding=1)
            ]),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            SEConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            ResBlock([
                SEConv2d(512, 512, kernel_size=3, stride=1, padding=1),
                SEConv2d(512, 512, kernel_size=3, stride=1, padding=1)
            ])
        )

        self.fm1_conf = nn.Conv2d(512, n_classes*self.fm1_priors, 3, padding=1)
        self.fm1_loc = nn.Conv2d(512, 4*self.fm1_priors, 3, padding=1)
        ## (512, 38, 38)

        self.fm2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            ResBlock([
                SEConv2d(512, 512, kernel_size=3, stride=1, padding=1),
                SEConv2d(512, 512, kernel_size=3, stride=1, padding=1),
                SEConv2d(512, 512, kernel_size=3, stride=1, padding=1)
            ]),
            nn.MaxPool2d(3, stride=1, padding=1),
            SEConv2d(512, 1024, kernel_size=3, stride=1, padding=6, dilation=6),
            ResBlock([
                SEConv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
            ])
        )
        self.fm2_conf = nn.Conv2d(1024, n_classes*self.fm2_priors, 3, padding=1)
        self.fm2_loc = nn.Conv2d(1024, 4*self.fm2_priors, 3, padding=1)
        ## (1024, 19, 19)

        self.fm3 = nn.Sequential(
            SEConv2d(1024, 256, kernel_size=1, padding=0),
            SEConv2d(256, 512, kernel_size=3, stride=2, padding=1),
        )
        self.fm3_conf = nn.Conv2d(512, n_classes*self.fm3_priors, 3, padding=1)
        self.fm3_loc = nn.Conv2d(512, 4*self.fm3_priors, 3, padding=1)
        ## (512, 10, 10)

        self.fm4 = nn.Sequential(
            SEConv2d(512, 128, kernel_size=1, padding=0),
            SEConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.fm4_conf = nn.Conv2d(256, n_classes*self.fm4_priors, 3, padding=1)
        self.fm4_loc = nn.Conv2d(256, 4*self.fm4_priors, 3, padding=1)
        ## (256, 5, 5)

        self.fm5 = nn.Sequential(
            SEConv2d(256, 128, kernel_size=1, padding=0),
            SEConv2d(128, 256, kernel_size=3, padding=0)
        )
        self.fm5_conf = nn.Conv2d(256, n_classes*self.fm5_priors, 3, padding=1)
        self.fm5_loc = nn.Conv2d(256, 4*self.fm5_priors, 3, padding=1)
        ## (256, 3, 3)

        self.fm6 = nn.Sequential(
            SEConv2d(256, 128, kernel_size=1, padding=0),
            SEConv2d(128, 256, kernel_size=3, padding=0)
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


class SEConv2d(nn.Module):

    def __init__(self, *arg, **kwargs):
        super(SEConv2d, self).__init__()

        if 'in_channels' in kwargs:
            inc = kwargs['in_channels']
        else:
            inc = arg[0]
        if 'out_channels' in kwargs:
            ouc = kwargs['out_channels']
        else:
            ouc = arg[1]

        self.conv = nn.Conv2d(*arg, **kwargs)
        self.bn = nn.BatchNorm2d(ouc)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(inc, inc)
        self.fc2 = nn.Linear(inc, ouc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        data = torch.mean(inp, dim=-1)
        data = torch.mean(data, dim=-1)
        # bs, channels
        data = self.relu(self.fc1(data))
        data = self.fc2(data)
        data = self.sigmoid(data)

        data2 = self.conv(inp)
        data2 = self.bn(data2)
        data2 = self.relu(data2)
        return data2*data[..., None, None]



class ResBlock(nn.Module):

    def __init__(self, nets):
        super(ResBlock, self).__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, data):
        out = data
        for m in self.nets:
            out = m(out)
        return out + data



        
