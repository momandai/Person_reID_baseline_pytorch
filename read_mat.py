import scipy.io as sio
import torch
from torchvision import models
# matfn = 'pytorch_result.mat'
# data = sio.loadmat(matfn)
# print('Information for mat4py.mat ')
# print(data)


model = models.SqueezeNet()

model2 = models.squeezenet1_1(pretrained=True)