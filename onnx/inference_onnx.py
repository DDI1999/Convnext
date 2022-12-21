import os
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from myTransforms import ResizeTo224
import time
from onnxruntime import InferenceSession


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)
    start = time.time()
    data_transform = transforms.Compose([
        ResizeTo224(),
        transforms.ToTensor(),
    ])
    # load image
    img_path = args.data_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img, cmap='gray')
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = '../class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r", encoding='utf-8') as f:
        class_indict = json.load(f)

    data_load_time = time.time()
    # load model weights
    weights_path = "E:/work/ConvNeXt/weights/best_conv_96.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    #     model.load_state_dict(torch.load(weights_path, map_location=device))
    model = torch.load(weights_path, map_location=device)

    # torch_prediction
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
    torch_time = time.time()

    # onnx_Inference
    onnx_model = InferenceSession('./best.onnx')
    input_name = onnx_model.get_inputs()[0].name
    t2 = time.time()
    result = onnx_model.run([], {input_name: np.array(img)})

    output_onnx = np.squeeze(result)
    predict_onnx = torch.softmax(torch.tensor(output_onnx), dim=0)
    onnx_time = time.time()

    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    print('-----------------------------')
    for i in range(len(predict_onnx)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict_onnx[i].numpy()))
    print('数据处理 {}'.format(data_load_time-start))
    print('torch {}'.format(torch_time-t1))
    print('onnx {}'.format(onnx_time-t2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../pre_img/mu.png")
    parser.add_argument('--num_classes', type=int, default=3)
    args = parser.parse_args()
    main(args)
