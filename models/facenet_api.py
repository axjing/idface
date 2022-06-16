import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.hub import load_state_dict_from_url
import matplotlib.pyplot as plt

from models.facenet import Facenet as facenet
from utils.common import img_normalization, img_resize, print_config


class Facenet(object):
    # 修改模型配置参数
    _defaults = {
        "model_path": "ckpt/facenet_mobilenet.pth",  # 训练好的模型存放路径,选择训练好的损失最低的权重文件
        "input_shape": [160, 160, 3],  # 输入图片的大小
        "backbone": "mobilenet",  # 主干特征提取网络
        "letterbox_image": True,  # 是否进行不失真的resize
        "cuda": True,  # 是否使用cuda
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()

        # print_config(**self._defaults)
        print("Activate The Face Recognition Algorithm...")

    def generate(self):
        """
        载入模型与权值
        :return:
        """

        print('=> Start loading the ({}) model...'.format(self.model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = facenet(backbone=self.backbone, mode="predict", pretrained=True).eval()
        if os.path.isfile(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
        else:
            name_model = os.path.split(self.model_path)[-1]
            state_dict = load_state_dict_from_url(
                "https://github.com/axjing/GenFaceID/releases/download/v0.001/" + name_model,
                model_dir="ckpt",
                progress=True)
        self.net.load_state_dict(state_dict, strict=False)
        print('=> model {} has been loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def detect_image(self, input_im, compared_im):
        """
        检测图片
        :param input_im:
        :param compared_im:
        :return:
        """

        #   图片预处理，归一化
        with torch.no_grad():
            input_im = img_resize(input_im, [self.input_shape[1], self.input_shape[0]],
                                  letterbox_image=self.letterbox_image)
            compared_im = img_resize(compared_im, [self.input_shape[1], self.input_shape[0]],
                                     letterbox_image=self.letterbox_image)

            photo_1 = torch.from_numpy(
                np.expand_dims(np.transpose(img_normalization(np.array(input_im, np.float32)), (2, 0, 1)), 0))
            photo_2 = torch.from_numpy(
                np.expand_dims(np.transpose(img_normalization(np.array(compared_im, np.float32)), (2, 0, 1)), 0))

            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            #   图片传入网络进行预测
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()

            #   计算二者之间的距离
            l1 = np.linalg.norm(output1 - output2, axis=1)

        return l1,input_im,compared_im
