
# -*-coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from PIL import Image

class FrozenPredict(object):
    """
    读取固化模型进行预测
    """

    def __init__(self, frozen_graph_path="weights/model.pb"):
        """
        读取配置
        """
        self.frozen_graph_path = frozen_graph_path
        if not os.path.exists(frozen_graph_path):
            raise RuntimeError("模型文件不存在！")
        self.load_model()
        self.input_x = self.graph.get_tensor_by_name('data:0')
        self.output_1 = self.graph.get_operation_by_name(
            'pool1/Mean').outputs[0]
        self.output_2 = self.graph.get_operation_by_name(
            'dense_1/BiasAdd').outputs[0]
        self.output_3 = self.graph.get_operation_by_name(
            'softmax_1/Softmax').outputs[0]

    def load_model(self):
        """
        加载模型
        """
        with open(self.frozen_graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # 导入计算图
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self.sess = tf.Session(graph=self.graph)

    def predict(self, input_x="img/1.jpg", ):
        """
        预测
        """
        tensor_name_list = [
            tensor.name for tensor in self.graph.as_graph_def().node][-10:]
        img = Image.open(input_x)
        img = img.resize((224, 224))
        Lrs = np.expand_dims(img, 0)
        feed_dict = {
            self.input_x: Lrs,
        }
        # sess.run获得模型的预测输出
        prediction = self.sess.run(
            [self.output_1, self.output_2, self.output_3], feed_dict=feed_dict)
        return prediction


if __name__ == "__main__":
    FrozenPredict().predict()
