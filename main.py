from os import name
from unittest.main import main
import keras
from keras import layers
from keras import activations
import numpy as np
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input
import os
import faiss
from keras import backend as K
from keras.preprocessing import image
import config


def get_model() -> keras.Model:
    """
    创建resnet中间层输出模型
    """
    resnet = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling='avg',
    )
    output = resnet.output
    output = keras.layers.Dense(64)(output)
    output = keras.layers.Softmax()(output)
    model = keras.Model(
        resnet.input, output
    )
    model.summary()

    model = K.function(
        inputs=[model.input],
        outputs=[
            model.layers[-3].output,
            model.layers[-2].output,
            model.layers[-1].output
        ])

    return model


def image_preproces(img_path: str):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_feature(model: keras.Model, img: np.array):
    '''
    ⽤模型提取⼀张图⽚的特征向量
    param:
    model 为上⾯定义的模型
    img 为⼀张图⽚
    return:
    fc_feat 为上⾯模型中定义的64维全连接层的输出
    conf_feat 为最后⼀层卷积层输出的特征图，并变形成⼀条⼀维向量
    classify 为上⾯模型的预测向量
    '''

    conv_feat, fc_feat, classify = model(img)
    return fc_feat, conv_feat, classify


def extract(model: keras.Model, img_folder: str, file_name: str):
    '''
    提取⼀个⽂件夹中的所有图⽚的特征，并以 .h5格式存储到指定位置
    param:
    model 为上⾯定义的模型
    img_folder 为⼀个⽂件夹路径
    file_name 为存储的⽂件地址
    return:
    ⽆返回
    '''
    import glob
    import h5py
    import tqdm
    img_list = glob.glob(os.path.join(img_folder, '*[jpg|png]'))
    name = []
    image = []
    for im in tqdm.tqdm(img_list):
        _, fc_feat, _ = model(image_preproces(im))
        image.append(fc_feat)
        name.append(os.path.basename(im))
    with h5py.File(file_name, 'w') as dataset:
        dataset.create_dataset('image', data=image)
        dataset.create_dataset('class_name', data=np.array(name, np.string_))
    return


def search(gallery, query) -> list:
    '''
    返回 gallery 中与 query 最近邻的3个结果
    param:
    gallery 为⼀个 faiss 对象
    query 为⼀张图⽚的 fc_feat向量
    return:
    res 为⼀个 list，对应前三个最近邻图⽚的名称
    '''
    res = None
    return res


def test_extract_teature():
    model = get_model()
    img = image_preproces('img/gen_00200.jpg')
    image_feature = extract_feature(model, img)
    print(image_feature)


if __name__ == "__main__":
    model = get_model()
    extract(model, 'img', 'image.h5')
