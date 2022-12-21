import os

import torch
import onnxruntime as rt
import torch.onnx
import cv2
import numpy as np
import onnx
from PIL import Image
from torchvision import transforms

from myTransforms import ResizeTo224


def test_onnx():
    data_transform = transforms.Compose([
        ResizeTo224(),
        transforms.ToTensor(),
    ])
    # load image
    img_path = "../pre_img/4919.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    img = np.array(img)
    sess = rt.InferenceSession("./convenxt_plume.onnx", None)

    # 打印输入节点的名字，以及输入节点的shape
    for i in range(len(sess.get_inputs())):
        print(sess.get_inputs()[i].name, sess.get_inputs()[i].shape)

    print("----------------")
    # 打印输出节点的名字，以及输出节点的shape
    for i in range(len(sess.get_outputs())):
        print(sess.get_outputs()[i].name, sess.get_outputs()[i].shape)

    input_name = sess.get_inputs()[0].name
    result = sess.run([], {input_name: img})
    print(result)


if __name__ == '__main__':
    test_onnx()
