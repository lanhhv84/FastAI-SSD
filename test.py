from ssd import MobileNetv2SSD
import torch


a = torch.FloatTensor(3, 3, 300, 300).uniform_(0, 1)
net = MobileNetv2SSD(2)
net.test(a)