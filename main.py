from os import name
from unittest.main import main
import h5py
from h5py._hl import dataset
import keras
import numpy as np
import os
import faiss
from classification_models.keras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')


def get_model() -> keras.Model:
    """
    创建resnet中间层输出模型
    """
    resnet = ResNet18(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3),
    )
    output = resnet.output
    output = keras.layers.GlobalAveragePooling2D(name='pool1')(output)
    output = keras.layers.Dense(64)(output)
    output = keras.layers.Softmax()(output)
    model = keras.Model(
        resnet.input, output
    )
    model.summary()
    model.save("weights/model.h5")
    return model


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
    from predict import FrozenPredict
    from PIL import Image
    predict = FrozenPredict().predict
    img_list = glob.glob(os.path.join(img_folder, '*[jpg|png]'))
    img_list.sort()

    name = []
    image = []
    for im in tqdm.tqdm(img_list):
        im = Image.open(im)
        im = im.resize((224, 224))
        im = np.expand_dims(im, 0)
        _, fc_feat, _ = predict(im)
        image.append(fc_feat[0])
        name.append(os.path.basename(im))
    with h5py.File(file_name, 'w') as dataset:
        dataset.create_dataset('image', data=np.array(image, np.float32))
        dataset.create_dataset('class_name', data=np.array(name, np.string_))
    return


def search(gallery, query, ) -> list:
    '''
    返回 gallery 中与 query 最近邻的3个结果
    param:
    gallery 为⼀个 faiss 对象
    query 为⼀张图⽚的 fc_feat向量
    return:
    res 为⼀个 list，对应前三个最近邻图⽚的名称
    '''
    import config
    index = faiss.IndexFlatL2(64)
    index.add(gallery)
    print(index.is_trained)
    D, I = index.search(query, 3)
    print(I)
    print(D)
    res = [[config.class_name[i] for i in j] for j in I]
    return res


