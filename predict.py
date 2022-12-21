import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from myTransforms import ResizeTo224


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

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
    json_path = 'E:/work/ConvNeXt/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    #     model = resnet34(num_classes=args.num_classes).to(device)
    #     net = resnet34()

    # change fc layer structure
    #     in_channel = net.fc.in_features
    #     net.fc = nn.Linear(in_channel, class_num)
    #     net.to(device)

    # load model weights
    weights_path = "weights/best_conv_96.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    #     model.load_state_dict(torch.load(weights_path, map_location=device))
    model = torch.load(weights_path, map_location=device)

    # prediction
    start = time.time()
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    end = time.time()
    print('time: {}'.format(end - start))

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./pre_img/mu.png")
    parser.add_argument('--num_classes', type=int, default=3)
    args = parser.parse_args()
    main(args)
