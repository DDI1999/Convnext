import sys
import os
import json
import logging as log

import torch
from openvino.inference_engine import IECore
import numpy as np
from PIL import Image

from torchvision import transforms

from myTransforms import ResizeTo224


# ie = IECore()
# # model="ctdet_coco_dlav0_512.onnx"
# model = "ctdet_coco_dlav0_512/ctdet_coco_dlav0_512.xml"
# net = ie.read_network(model=model)
# input_blob = next(iter(net.input_info))
# out_blob = next(iter(net.outputs))
# net.batch_size = 16  # batchsize
#
# n, c, h, w = net.input_info[input_blob].input_data.shape
# print(n, c, h, w)
# images = np.ndarray(shape=(n, c, h, w))
# for i in range(n):
#     image = cv2.imread("123.jpg")
#     if image.shape[:-1] != (h, w):
#         image = cv2.resize(image, (w, h))
#     image = image.transpose((2, 0, 1))
#     images[i] = image
# exec_net = ie.load_network(network=net, device_name="CPU")
# start = time.time()
# res = exec_net.infer(inputs={input_blob: images})
# # print(res)
# print('infer total time is %.4f s' % (time.time() - start))


def main():
    device = "CPU"
    model_xml_path = "../IR/32/resNet34.xml"
    model_bin_path = "../IR/32/resNet34.bin"
    img_path = '../test_img/test_mu.PNG'

    # set log format
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    assert os.path.exists(model_xml_path), ".xml file does not exist..."
    assert os.path.exists(model_bin_path), ".bin file does not exist..."

    # search *.jpg files
    # image_list = glob.glob(os.path.join(image_path, '*', "*.jpg"))
    # assert len(image_list) > 0, "no image(.jpg) be found..."

    # inference engine
    ie = IECore()

    # read IR
    net = ie.read_network(model=model_xml_path, weights=model_bin_path)
    # load model
    exec_net = ie.load_network(network=net, device_name=device)

    # get input and output name
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))

    # set batch size
    batch_size = 1
    net.batch_size = batch_size

    # read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    # images = np.ndarray(shape=(n, c, h, w))
    # inference every image
    data_transform = transforms.Compose([
        ResizeTo224(),
        transforms.ToTensor(),
    ])
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # start sync inference
    res = exec_net.infer(inputs={input_blob: img})
    prediction = np.squeeze(res[output_blob])
    predict = torch.softmax(torch.tensor(prediction), dim=0)

    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))

    # # np softmax process
    # prediction -= np.max(prediction, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大元素
    # prediction = np.exp(prediction) / np.sum(np.exp(prediction), keepdims=True)
    # class_index = np.argmax(prediction, axis=0)
    # print("prediction: '{}'\nclass:{}  probability:{}\n".format(img,
    #                                                             class_indict[str(class_index)],
    #                                                             np.around(prediction[class_index]), 2))


if __name__ == '__main__':
    main()
