import glob
import os
import numpy as np
import PIL.Image as Image


# 在训练集中随机生成512张各类别数量比例不变的矫正数据
# root_dir=./train
def random_img(root_dir='./data_224_L/train', nums=512):
    save_s_num = 0
    all_nums = len(glob.glob(os.path.join(root_dir, '*', '*.jpg')))  # train下所有图片
    save_path = './calib_train_data/train'  # 保存路径

    pechs = 0
    for path in os.listdir(root_dir):
        if pechs < len([o for o in os.listdir(root_dir)]) - 1:
            pechs += 1
            os.makedirs(os.path.join(save_path, path), exist_ok=True)
            path_nums = len(glob.glob(os.path.join(root_dir, path, '*.jpg')))
            ratio = (path_nums / all_nums)
            files = glob.glob(os.path.join(root_dir, path, '*.jpg'))
            save_s_num += int(nums * ratio)
            # 打乱每个类文件夹的图片顺序
            np.random.shuffle(files)

            for i, file in enumerate(files):
                img = Image.open(file)
                assert img.mode == 'L'
                if i < int(nums * ratio):
                    img.save(os.path.join(save_path, path, file.split('\\')[-1].split('.')[0] + '.jpg'))
        else:
            os.makedirs(os.path.join(save_path, path), exist_ok=True)
            files = glob.glob(os.path.join(root_dir, path, '*.jpg'))
            s_num = nums - int(save_s_num)
            # 打乱每个类文件夹的图片顺序
            np.random.shuffle(files)

            for i, file in enumerate(files):
                img = Image.open(file)
                assert img.mode == 'L'
                if i < int(s_num):
                    img.save(os.path.join(save_path, path, file.split('\\')[-1].split('.')[0] + '.jpg'))

    lss = []
    for j in os.listdir(save_path):
        lss += (glob.glob(os.path.join(save_path, j, '*.jpg')))
    print(len(lss))


if __name__ == '__main__':
    random_img('./data_224_L/train', 512)
