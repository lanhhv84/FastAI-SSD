import torch
import torch.nn as nn
import numpy as np


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
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def create_conv(*arg, **kwargs):
        return nn.Sequential(
            nn.Conv2d(*arg, **kwargs), 
            nn.BatchNorm2d(arg[1]), 
            nn.ReLU()
        )

    def forward(self, data):
        # assert data.min() >= -1.1, "Expect {} to larger than 0".format(data.min())
        # assert data.max() <= 1.1, "Expect {} to smaller than 1".format(data.max())
        fm = data
        conf = []
        loc = []
        for layer, layer_conf, layer_loc in zip(self.fms, self.fms_conf, self.fms_loc):
            fm = layer(fm)
            conf.append(layer_conf(fm))
            loc.append(self.sigmoid(layer_loc(fm)))
        return loc, conf
    
    def test(self, inp):
        bs = inp.shape[0]

        loc_shape = [[bs, 4*self.fm1_priors, 38, 38], \
            [bs, 4*self.fm2_priors, 19, 19], \
                [bs, 4*self.fm3_priors, 10, 10], \
                    [bs, 4*self.fm4_priors, 5, 5], \
                        [bs, 4*self.fm5_priors, 3, 3], \
                            [bs, 4*self.fm6_priors, 1, 1]
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
    """
    Single shot multibox object detection with MobileNetSSD
    """

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
            nn.ReLU(),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(),
        )

    


def iou_table(x, truth, bs, nboxes, maxlen, im_size, eps = 1e-3):
    pred_xA = (x[:, 0, :] - x[:, 2, :]/2).contiguous().view(bs, nboxes)[..., None].expand(bs, nboxes, maxlen)
    pred_yA = (x[:, 1, :] - x[:, 3, :]/2).contiguous().view(bs, nboxes)[..., None].expand(bs, nboxes, maxlen)
    pred_xB = (x[:, 0, :] + x[:, 2, :]/2).contiguous().view(bs, nboxes)[..., None].expand(bs, nboxes, maxlen)
    pred_yB = (x[:, 1, :] + x[:, 3, :]/2).contiguous().view(bs, nboxes)[..., None].expand(bs, nboxes, maxlen)

    truth_xA = (truth[:, 0, :]).contiguous().view(bs, maxlen)[:, None, :].expand(bs, nboxes, maxlen)
    truth_yA = (truth[:, 1, :]).contiguous().view(bs, maxlen)[:, None, :].expand(bs, nboxes, maxlen)
    truth_xB = (truth[:, 2, :]).contiguous().view(bs, maxlen)[:, None, :].expand(bs, nboxes, maxlen)
    truth_yB = (truth[:, 3, :]).contiguous().view(bs, maxlen)[:, None, :].expand(bs, nboxes, maxlen)




    xA = torch.max(pred_xA, truth_xA)
    yA = torch.max(pred_yA, truth_yA)
    xB = torch.min(pred_xB, truth_xB)
    yB = torch.min(pred_yB, truth_yB)
       
        
        
    interX = torch.clamp(xB - xA + 1/im_size, min=0)
    interY = torch.clamp(yB - yA + 1/im_size, min=0)
    interArea =  interX* interY
    

    boxAArea = (pred_xB - pred_xA + 1/im_size) * (pred_yB - pred_yA + 1/im_size)
    boxBArea = (truth_xB - truth_xB + 1/im_size) * (truth_yB - truth_yA + 1/im_size)
    ious = interArea / torch.clamp(boxAArea + boxBArea - interArea, min=eps) # [bs, 8732, maxlen]
    return ious



def ssd_loss(out, truth_loc, truth_conf, smoothl1, cre, device, n_classes, im_size, iou_thres=0.6, eps = 1e-3):
    loc, conf = out
    # print(max([x.max().item() for x in loc]), min([x.min().item() for x in loc]), max([x.max().item() for x in conf]), min([x.min().item() for x in conf]))
    assert min([x.min().item() for x in loc]) < 0.999
    assert len(loc) == 6
    assert len(conf) == 6
    assert len(loc[0].shape) == 4 
    bs = truth_loc.shape[0]
    maxlen = truth_loc.shape[1]
    # offset (calculate center)
    truth = (truth_loc.permute(0, 2, 1) + 1)*(im_size/2)
    truth[:, 0, :], truth[:, 1 ,:] = truth[:, 1 ,:], truth[:, 0 ,:] 
    truth[:, 2, :], truth[:, 3, :] = truth[:, 3, :] - truth[:, 0, :], truth[:, 2, :] - truth[:, 1 ,:]
    truth[:, 0, :], truth[:, 1 ,:] = truth[:, 0, :] + 0.5*truth[:, 2, :], truth[:, 1 ,:] + 0.5*truth[:, 3, :]   # cx, cy, w, h
    
    
    truth = truth/im_size
    
    
    for i in range(len(loc)):
        # assert torch.sum(torch.isnan(loc[i])) == 0, "loc is NaN"
        loc[i] = loc[i].reshape(bs, -1, 4, loc[i].shape[-1], loc[i].shape[-1])
        conf[i] = conf[i].reshape(bs, -1, n_classes, loc[i].shape[-1], loc[i].shape[-1])
        
    """
        loc:
            - (4, 4, 38, 38)
            - (6, 4, 19, 19)
            - (6, 4, 10, 10)
            - (6, 4, 5, 5)
            - (4, 4, 3, 3)
            - (4, 4, 1, 1)
            
        conf:
            - (4, n_classes, 38, 38)
            - (6, n_classes, 19, 19)
            - (6, n_classes, 10, 10)
            - (6, n_classes, 5, 5)
            - (4, n_classes, 3, 3)
            - (4, n_classes, 1, 1)
        truth_loc: (bs, maxlen, 4)
        truth_conf: (bs, maxlen)
    """
    
    
    ratios4 = [1, 2, 0.5, 1]
    ratios6 = [1, 2, 0.5, 3, 1/3, 1]
    
    ratios_map = [ ratios4, ratios6, ratios6, ratios6, ratios4, ratios4]
    scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
    
    pred = []
    priors = []
    
    for index in range(len(loc)):
        prior = torch.zeros_like(loc[index]) # bs, -1, 4, loc[i].shape[-1], loc[i].shape[-1]
        # calculate prior (cx, cy)
        for row in range(prior.shape[-1]):
            for col in range(prior.shape[-1]):
                prior[..., 0, row, col] = (0.5 + row) / prior.shape[-1] # Prior cx
                prior[..., 1, row, col] = (0.5 + col) / prior.shape[-1] # prior cy
        # assert prior.max() <= 1
        # calculate prior (w, h)
        ratios = ratios_map[index]
        scale = scales[index]
        for i in range(len(ratios)):
            r = ratios[i]
            if i == len(ratios) - 1:
                scale = scale*1.5
            w = scale*np.sqrt(r)
            h = scale/np.sqrt(r)
            # w
            prior[..., i, 2, :, :]  = w/prior.shape[-1]
            prior[..., i, 3, :, :] = h/prior.shape[-1]
        # done prior
        # cx
        # assert torch.sum(torch.isnan(prior)) == 0, "prior is NaN"
        pred_box = torch.zeros_like(loc[index])
        pred_box[...,0,:,:] = loc[index][...,0,:,:]*prior[...,2,:,:] + prior[...,0,:,:]
        pred_box[...,1,:,:] = loc[index][...,1,:,:]*prior[...,3,:,:] + prior[...,1,:,:]
        pred_box[...,2,:,:] = torch.exp(loc[index][...,2,:,:])*prior[...,2,:,:]
        pred_box[...,3,:,:] = torch.exp(loc[index][...,3,:,:])*prior[...,3,:,:]
        priors.append(prior)
        pred_box = pred_box.permute(0, 2, 1, 3, 4).contiguous().view(bs, 4, -1)
        pred_box = torch.clamp(pred_box, min=0, max=1)
        # assert torch.sum(torch.isnan(pred_box)) == 0, "Pred is NaN"
        pred.append(pred_box)
    
    pred = torch.cat(pred, dim=2) # (bs, 4, 8732)
    assert list(pred.shape) == [bs, 4, 8732]
    
    assert list(truth.shape) == [bs, 4, maxlen]
    # IoU
    ious = iou_table(pred, truth=truth, bs=bs, nboxes=pred.shape[-1], im_size=im_size, maxlen=maxlen)
    pconf = torch.cat([c.permute(0, 1, 3, 4, 2).contiguous().view(bs, -1, n_classes) for c in conf], dim=1)
    assert list(pconf.shape) == [bs, 8732, n_classes]
    max_ious, max_match = torch.max(ious, dim=2)
    positive_match = max_ious > iou_thres # (bs, 8732)
    assert list(positive_match.shape) == [bs, 8732]
    pos_pred = pred.permute(0, 2, 1)[positive_match] # (total_match in batch, 4)
    # done
    # 443 -> 4, 4, 3, 8732
    tr = truth[..., None].expand(*truth.shape, 8732).permute(0, 3, 2, 1) # 4, 8732, 3, 4
    assert list(tr.shape) == [bs, 8732, maxlen, 4]
    
    
    truth_mask = torch.zeros_like(ious).byte().view(-1, maxlen) # (bs, 8732, maxlen)
    truth_mask.scatter_(1, max_match.view(-1)[:, None], 1.)
    truth_mask = truth_mask.view(bs, -1, maxlen)
    
    # assert torch.sum(truth_mask).item() == 8732*bs, "Total number of item is {}".format(torch.sum(truth_mask).item())
    
    # pass
    truth_mask = positive_match[..., None]*truth_mask
    tr = tr[truth_mask]
    
    
    assert list(tr.shape) == list(pos_pred.shape)
    loc_loss = smoothl1(pos_pred, tr)
    # confidence loss
    """
    
    
    """
    assert list(pconf.shape) == [bs, 8732, n_classes]
    
    
    
    priors = torch.cat([c.permute(0, 1, 3, 4, 2).contiguous().view(bs, -1, 4) for c in priors], dim=1)
    assert list(priors.shape) == [bs, 8732, 4]
    priors = priors.permute(0, 2, 1)
    truth_ious = iou_table(priors, truth=truth, bs=bs, im_size=im_size, nboxes=pred.shape[-1], maxlen=maxlen)

    max_iou, max_index = torch.max(truth_ious, dim=-1)# [bs, 8732], [bs, 8732]
    # index: (bs, 8732) value (bs, 8732, maxlen)
    truth_conf = truth_conf[:, None, :].expand(bs, 8732, maxlen) # (bs, 8732, maxlen)
    assert list(truth_conf.shape) == [bs, 8732, maxlen]
    tcof = truth_conf.gather(2, max_index[..., None].to(device))
    tcof[max_iou < iou_thres] = 0
    assert list(tcof.shape) == [bs, 8732, 1], "Got shape {}".format(list(tcof.shape))
    """
    Use truth_ious as a mask
    """
    
    tcof = tcof.view(bs, -1)
    
    npos = torch.clamp(torch.sum(tcof > 0), min=1e-3)
    nneg = 3*npos
    
    pos_index = tcof > 0
    neg_index = tcof == 0
    if npos > eps:
        conf_pos_loss = cre(pconf[pos_index].view(-1, n_classes), tcof[pos_index].view(-1))
        # assert torch.sum(torch.isnan(conf_pos_loss)) == 0, "Positive confidence loss is Nan"
    else:
        conf_pos_loss = 0
    if nneg >= 1:
        
        p = pconf[neg_index].view(-1, n_classes)
        t = tcof[neg_index].view(-1, 1)
        
        maxp = torch.max(p, dim=1)[0]
        
        max_error = torch.topk(maxp, int(nneg.item()), dim=0)[1]
        p = p[max_error, :]
        t = t[max_error, :]
        
        conf_neg_loss = cre(p, t.view(-1))
        # assert torch.sum(torch.isnan(conf_neg_loss)) == 0, "Negative confidence loss is Nan"
    else:
        conf_neg_loss = 0
    
    conf_loss = conf_pos_loss + conf_neg_loss
    # assert torch.sum(torch.isnan(loc_loss)) == 0, "Localization loss is NaN"
    npos_loc =  torch.clamp(torch.sum(truth_mask).float(), min=eps)
    npos_conf = torch.clamp(npos.float(), min=eps)
    
    return loc_loss/npos_loc + conf_loss/npos_conf


class MobileNetv2SSD(SSD):
    """
    Single shot multibox object detection with MobileNetSSD
    """

    def __init__(self, n_classes=2):
        super(MobileNetv2SSD, self).__init__(n_classes=n_classes)

        self.fm1 = nn.Sequential(
            MobileNetv2SSD.conv_dw(3, 64, 1, 1),
            MobileNetv2SSD.conv_dw(64, 64, 1, 1),
            nn.MaxPool2d(2, stride=2),

            MobileNetv2SSD.conv_dw(64, 128, 1, 1),
            MobileNetv2SSD.conv_dw(128, 128, 1, 1),
            nn.MaxPool2d(2, stride=2),

            MobileNetv2SSD.conv_dw(128, 256, 1, 1),
            MobileNetv2SSD.conv_dw(256, 256, 1, 1),
            MobileNetv2SSD.conv_dw(256, 256, 1, 1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            MobileNetv2SSD.conv_dw(256, 512, 1, 1),
            MobileNetv2SSD.conv_dw(512, 512, 1, 1),
            MobileNetv2SSD.conv_dw(512, 512, 1, 1)
        )
        ## (512, 38, 38)
        self.fm1_conf = nn.Conv2d(512, n_classes*self.fm1_priors, 3, padding=1)
        self.fm1_loc = nn.Conv2d(512, 4*self.fm1_priors, 3, padding=1)
        

        self.fm2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            MobileNetv2SSD.conv_dw(512, 512, 1, 1),
            MobileNetv2SSD.conv_dw(512, 512, 1, 1),
            MobileNetv2SSD.conv_dw(512, 512, 1, 1),
            nn.MaxPool2d(3, stride=1, padding=1),
            MobileNetv2SSD.conv_dw(512, 1024, 1, 6, 6),
            MobileNetv2SSD.create_conv(1024, 1024, 1, stride=1, padding=0)
        )
        self.fm2_conf = nn.Conv2d(1024, n_classes*self.fm2_priors, 3, padding=1)
        self.fm2_loc = nn.Conv2d(1024, 4*self.fm2_priors, 3, padding=1)
        ## (1024, 19, 19)

        self.fm3 = nn.Sequential(
            MobileNetv2SSD.create_conv(1024, 256, 1, padding=0),
            MobileNetv2SSD.conv_dw(256, 512, 2, 1),
        )
        self.fm3_conf = nn.Conv2d(512, n_classes*self.fm3_priors, 3, padding=1)
        self.fm3_loc = nn.Conv2d(512, 4*self.fm3_priors, 3, padding=1)
        ## (512, 10, 10)

        self.fm4 = nn.Sequential(
            MobileNetv2SSD.create_conv(512, 128, 1, padding=0),
            MobileNetv2SSD.conv_dw(128, 256, 2, 1)
        )
        self.fm4_conf = nn.Conv2d(256, n_classes*self.fm4_priors, 3, padding=1)
        self.fm4_loc = nn.Conv2d(256, 4*self.fm4_priors, 3, padding=1)
        ## (256, 5, 5)

        self.fm5 = nn.Sequential(
            MobileNetv2SSD.create_conv(256, 128, 1, padding=0),
            MobileNetv2SSD.conv_dw(128, 256, 1, 0)
        )
        self.fm5_conf = nn.Conv2d(256, n_classes*self.fm5_priors, 3, padding=1)
        self.fm5_loc = nn.Conv2d(256, 4*self.fm5_priors, 3, padding=1)
        ## (256, 3, 3)

        self.fm6 = nn.Sequential(
            MobileNetv2SSD.create_conv(256, 128, 1, padding=0),
            MobileNetv2SSD.conv_dw(128, 256, 1, 0)
        )
        self.fm6_conf = nn.Conv2d(256, n_classes*self.fm6_priors, 3, padding=1)
        self.fm6_loc = nn.Conv2d(256, 4*self.fm6_priors, 3, padding=1)
        ## (256, 1, 1)

        self.fms = [self.fm1, self.fm2, self.fm3, self.fm4, self.fm5, self.fm6]
        self.fms_conf = [self.fm1_conf, self.fm2_conf, self.fm3_conf, self.fm4_conf, self.fm5_conf, self.fm6_conf]
        self.fms_loc = [self.fm1_loc, self.fm2_loc, self.fm3_loc, self.fm4_loc, self.fm5_loc, self.fm6_loc]

    @staticmethod
    def conv_dw(inp, oup, stride, pad, dil=1):
        return MobileNetv2Block(inp=inp, oup=oup, stride=stride, pad=pad, dil=dil)

class MobileNetv2Block(nn.Module):

    def __init__(self, inp, oup, stride, pad, dil=1):
        super(MobileNetv2Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp, inp*6, 1, stride=1, padding=0),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp*6, inp*6, 3, stride=stride, padding=pad, dilation=dil),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp*6, oup, 1, stride=1, padding=0)
        )

    def forward(self, data):
        out = self.block(data)
        if list(out.shape) == list(data.shape):
            return out + data
        return out
