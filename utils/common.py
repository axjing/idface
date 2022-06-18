import os
import time
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

def show_two_image(input_im,compared_im,ins=None,wait=0):
    fig=plt.figure(figsize=(9,6))
    plt.ion()    # 打开交互模式
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(np.array(input_im))
    plt.subplot(1, 2, 2)
    plt.title("FaceDB Image")
    plt.imshow(np.array(compared_im))
    plt.text(-20, -20, 'Distance:{:.3f}'.format(ins), ha='center', va='bottom', fontsize=14, color="r")
    fig.show()
    # # 显示前关掉交互模式
    plt.ioff()
    # plt.pause(wait)                     # 显示秒数
    plt.clf()
    plt.close(fig)
    return fig

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def img_resize(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def img_normalization(image):
    image /= 255.0 
    return image

def print_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def get_file_path(dir,suffix="jpg"):

    path_list=[]
    for f in os.listdir(dir):
        if f.split(".")[-1]==suffix:
            path_list.append(os.path.join(dir,f))

    return path_list

def get_filename_in_path(file_path:str):
    return os.path.split(file_path)[-1].split(".")

