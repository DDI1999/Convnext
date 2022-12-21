import os

import torch
from PIL import Image
from matplotlib import pyplot as plt

from convnext import convnext_tiny
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from torchvision import transforms

from myTransforms import ResizeTo224

data_transform = transforms.Compose([
    ResizeTo224(),
    transforms.ToTensor(),
])

# load image
img_path = "./pre_img/mu.png"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path)
plt.imshow(img, cmap='gray')
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
model = convnext_tiny(in_chans=1, num_classes=2).to("cpu")
model_status_dict = torch.load("./weights/best_conv_dict_96.pth", map_location="cpu")
model.state_dict(model_status_dict)
model.eval()
model.cpu()

# print(model._modules)
# print(model._modules['downsample_layers'])
# for i in model.parameters():
#     print(i)
# print("--------------------------------")
# for name, par in model.named_parameters():
#     print(name)

# print(model._modules["stages"][3][2].dwconv)

target_layers = [model._modules["stages"][3][2].dwconv]

cam = EigenCAM(model, target_layers, use_cuda=False)
grayscale_cam = cam(img)[0, :, :]
cam_image = show_cam_on_image(img.numpy(), grayscale_cam, use_rgb=True)
img = Image.fromarray(cam_image)
img.show()
