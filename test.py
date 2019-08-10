from ssd import MobileNetv2SSD
import torch


a = torch.zeros(3, 3, 300, 300)
net = MobileNetv2SSD(2)
net.test(a)