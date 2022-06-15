# GenFaceID
通用的人脸识别技术

## 目录

[TOC]

## 安装环境
```
pip install -r requirements.txt
```

## 测试步骤
### 使用预训练权重
1. 运行`test2img.py`：
```shell
python test2img.py
```

## 训练步骤
1. 本文使用如下格式进行训练。
```
|-datasets
    |-people0
        |-123.jpg
        |-234.jpg
    |-people1
        |-345.jpg
        |-456.jpg
    |-...
```  
2. 下载好数据集，将训练用的CASIA-WebFaces数据集以及评估用的LFW数据集，解压后放在根目录。
3. 在训练前利用`generate_annotation_file.py`文件生成对应的cls_train.txt。  
4. 利用`train.py`训练facenet模型，训练前，根据自己的需要选择backbone，model_path和backbone一定要对应。
5. 运行`train.py`即可开始训练。

## 评估步骤
1. 下载好评估数据集，将评估用的LFW数据集，解压后放在根目录
2. 在eval_LFW.py设置使用的主干特征提取网络和网络权值。
3. 运行eval_LFW.py来进行模型准确率评估。

## Reference
1. https://github.com/davidsandberg/facenet
2. https://github.com/timesler/facenet-pytorch
3. https://github.com/bubbliiiing/facenet-retinaface-pytorch

