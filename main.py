from os import name
from unittest.main import main
import h5py
from h5py._hl import dataset
import keras
from keras import layers
from keras import activations
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input
import os
import faiss
from keras import backend as K
from keras.preprocessing import image
import config
from PIL import Image


def get_model() -> keras.Model:
    """
    创建resnet中间层输出模型
    """
    resnet = VGG16(
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
    """
    处理输入图像
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def image_preproces_v2(img):
    x = Image.open(img).resize((224, 224))
    print(x.size)
    x = np.array(x, np.float32)
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

    conv_feat, fc_feat, classify = model([img])
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
        _, fc_feat, _ = model([image_preproces(im)])
        image.append(fc_feat[0])
        name.append(os.path.basename(im))
    with h5py.File(file_name, 'w') as dataset:
        dataset.create_dataset('image', data=np.array(image, np.float32))
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
    index = faiss.IndexFlatL2(64)
    index.add(gallery)
    print(index.is_trained)
    D, I = index.search(query, 3)
    print(I)
    print(D)
    res = [[config.class_name[i] for i in j] for j in I]
    return res


def test_extract_teature():
    """
    测试函数 extract_feature
    """
    model = get_model()
    img = image_preproces('img/gen_00200.jpg')
    image_feature = extract_feature(model, img)
    print(image_feature)


def test_get_h5():
    """
    测试函数 get_model
    """
    model = get_model()
    extract(model, 'img', 'image.h5')


if __name__ == "__main__":
    # test_get_h5()
    with h5py.File('image.h5', 'r') as dataset:
        images = dataset['image'][:]
        name = dataset['class_name']
        print([i for i in name])
    model = get_model()
    img = image_preproces('img/gen_00400.jpg')
    image_feature, _, _ = extract_feature(model, img)
    image_feature = np.array(image_feature, np.float32)
    res = search(images, image_feature)
    print(res)
