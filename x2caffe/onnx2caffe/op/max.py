# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/25 10:55
@Auth ： 郭吴畏
@File ：max.py
@IDE ：PyCharm
@Motto:keep moving forward, no one will stop waiting for you
"""
import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Max(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Max')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0] + self.inputs_buf[1])
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 2 # Caffe Eltwise SUM
            self.attrs = self.eltwise_param
            self.setParsed()
        # elif (self.inputs_buf[0] is not None or self.inputs_buf[1] is not None) or (self.inputs_shape[0] != self.inputs_shape[1]):
        #     self.type = 'Bias'
        #
        #     if self.inputs_shape[0] is None or self.inputs_shape[1] is None:
        #         self.unSupported('Inputs incompatible shapes for Caffe. ' + str(self.inputs_shape[0]) + ' + ' + str(self.inputs_shape[1]))
        #         return
        #
        #     inputs_size0 = np.multiply.reduce(self.inputs_shape[0], axis=None) if len(self.inputs_shape[0]) > 0 else 0
        #     inputs_size1 = np.multiply.reduce(self.inputs_shape[1], axis=None) if len(self.inputs_shape[1]) > 0 else 0
        #
        #     if self.inputs_buf[0] is not None and self.inputs_buf[1] is None:
        #         self.inputs.reverse()
        #         self.inputs_shape.reverse()
        #         self.inputs_buf.reverse()
        #     elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and inputs_size0 < inputs_size1:
        #         self.inputs.reverse()
        #         self.inputs_shape.reverse()
        #         self.inputs_buf.reverse()
        #
        #     if self.inputs_buf[1] is not None and np.all(self.inputs_buf[1] == 0):
        #         self.byPassOperator()
        #         return
        #
        #     BiasShape = self.inputs_shape[1]
        #     CompatibleFlag = self.checkShapeCompatible()
        #     if CompatibleFlag == 'Squeeze':
        #         self.type = 'Reshape+Bias'
        #     elif not CompatibleFlag:
        #         self.unSupported('Inputs incompatible shape for Caffe. ' + str(self.inputs_shape[0]) + ' + ' + str(BiasShape))
        #         return
        #
        #     self.bias_param = dict()
        #     self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
        #
        #     if self.inputs_buf[1] is not None:
        #         self.bias_param['num_axes'] = len(self.inputs_buf[1].shape) if isinstance(self.inputs_buf[1], np.ndarray) else 0
        #     else:
        #         self.bias_param['num_axes'] = len(self.inputs_shape[1])
        #
        #     self.attrs = self.bias_param
        #
        #     self.bias = self.inputs_buf[1]
        #
        #     self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))

        self.setConverted()

        return layers
