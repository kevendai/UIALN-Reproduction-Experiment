# DataLoader
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

def get_transform():
    transform =  transforms.Compose([
        # RGB转化为LAB
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
        # 只保留L通道
        transforms.Lambda(lambda x: x[:, :, 0]),
        transforms.ToTensor(),
    ])
    return transform



# 分解数据集
'''class retinex_decomposition_data(Dataset):
    def __init__(self, I_no_light, I_light):
'''


if __name__ == "__main__":
    transform = get_transform()
    img = cv2.imread("dataset\UIALN_datasest\Synthetic_dataset\synthetic dataset with outdoor image\synthetic dataset no artificial light\type I\0_B_0.779_d_0.01-10.01_D_7.9451.bmp", cv2.IMREAD_COLOR)
    print(img.shape)
    img = transform(img)
    print(img.shape)

