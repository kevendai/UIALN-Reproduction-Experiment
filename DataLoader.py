# DataLoader
import cv2
import numpy as np
import torch.random
from torch.utils.data import Dataset
from torchvision import transforms
import os

def get_transform_0():
    transform =  transforms.Compose([
        # RGB转化为LAB
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
        # 只保留L通道
        transforms.Lambda(lambda x: x[:, :, 0]),
        transforms.ToTensor(),
    ])
    return transform



# 分解数据集
class retinex_decomposition_data(Dataset):
    def __init__(self, I_no_light_path, I_light_path):
        self.I_light_imglist = self.get_path(I_light_path)
        self.I_no_light_imglist = [os.path.join(I_no_light_path, os.path.basename(img_path)) for img_path in self.I_light_imglist]
        self.transform = get_transform_0()

    def get_path(self, path):
        img_name_list = sorted(os.listdir(path))
        img_list = []
        for img_name in img_name_list:
            img_list.append(os.path.join(path, img_name))
        return img_list

    def __len__(self):
        return len(self.I_no_light_imglist)

    def __getitem__(self, index):
        I_no_AL_img_path = self.I_no_light_imglist[index]
        I_AL_img_path = self.I_light_imglist[index]

        I_no_AL_img = cv2.imread(I_no_AL_img_path, cv2.IMREAD_COLOR)
        I_AL_img = cv2.imread(I_AL_img_path, cv2.IMREAD_COLOR)

        # 检查图片是否读取成功
        if I_no_AL_img is None or I_AL_img is None:
            print(index)
            print(I_AL_img_path)
            print(I_AL_img)
            print("Error: 图片读取失败")
            exit(0)

        I_no_AL_img = cv2.cvtColor(I_no_AL_img, cv2.COLOR_BGR2RGB)
        I_AL_img = cv2.cvtColor(I_AL_img, cv2.COLOR_BGR2RGB)

        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        I_no_AL_tensor = self.transform(I_no_AL_img)
        torch.random.manual_seed(seed)
        I_AL_tensor = self.transform(I_AL_img)

        return I_no_AL_tensor, I_AL_tensor


if __name__ == "__main__":
    # 使用retinex_decomposition_data(Dataset)随机取出一组查看
    I_no_light_path = r"./dataset/UIALN_datasest/train_data/dataset_no_AL"
    I_light_path = r"./dataset/UIALN_datasest/train_data/dataset_with_AL/train"
    dataset = retinex_decomposition_data(I_no_light_path, I_light_path)
    I_no_AL_tensor, I_AL_tensor = dataset[50]
    # 转化为image
    I_no_AL_img = transforms.ToPILImage()(I_no_AL_tensor)
    I_AL_img = transforms.ToPILImage()(I_AL_tensor)
    I_no_AL_img.show()
    I_AL_img.show()

