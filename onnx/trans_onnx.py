import os

import torch
from torch import nn

# 将torch的 .pt 转换成 onnx
import convnext
from model import resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.load('filename.pth').to(device)

# model = convnext.convnext_tiny(num_classes=3).to(device)
# model = resnet34()
# in_channel = model.fc.in_features
# model.fc = nn.Linear(in_channel, 3)
# model.to(device)
model = torch.load('E:/work/ConvNeXt/weights/best_conv_96.pth', map_location=device)
# model.load_state_dict(torch.load('../weights/resNet34_plume_rgb_224.pth', map_location=device))

# weights_path = "../weights/resnet_plume_dict_224.pth"
# assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
# #     model.load_state_dict(torch.load(weights_path, map_location=device))
# model = torch.load(weights_path, map_location=device)

model.eval()
batch_size = 1  # 批处理大小
input_shape = (1, 224, 224)  # 输入数据

input_data_shape = torch.randn(batch_size, *input_shape, device=device)

torch.onnx.export(model, input_data_shape, "best.onnx", verbose=True)
