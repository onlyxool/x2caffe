import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Div(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Div')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.node.output[0], self.inputs_buf[0] / self.inputs_buf[1])
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is not None:
            self.type = 'Scale'

            # Scale Parameter
            self.scale_param = dict()
            self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if np.ones(self.inputs_shape[1]).size > 1 else 0
            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.scale_param['bias_term'] = False

            # Weight & Bias
            self.weight = 1/self.inputs_buf[1]
            self.bias = None

            self.attrs = self.scale_param

            self.setParsed()
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.type = 'Power+Scale'

            self.power_param = dict()
            self.power_param['power'] = -1
            self.power_param['scale'] = 1
            self.power_param['shift'] = 0

            WeightShape = self.inputs_shape[1]
            CompatibleFlag = self.checkShapeCompatible()

            if CompatibleFlag == 'Squeeze':
                self.type = 'Power+Reshape+Scale'
                self.inputs_shape[1] = [1] if self.inputs_shape[1] == [] else self.inputs_shape[1]
            elif not CompatibleFlag:
                self.unSupported('Inputs incompatible shapes for Caffe. ' + str(self.inputs_shape[0]) + ' x ' + str(WeightShape))
                return

            self.scale_param = dict()
            self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if np.ones(self.inputs_shape[1]).size > 1 else 0
            self.scale_param['num_axes'] = len(self.inputs_shape[1]) if np.ones(self.inputs_shape[1]).size > 1 else 0
            self.scale_param['bias_term'] = False

            self.weight = None

            self.attrs = self.scale_param

            self.setParsed()
        else:
            self.unSupported('Can\'t Support Operand[1] == {}.'.format(self.inputs_buf[1]))


    def convert(self):
        layers = list()
        if self.type == 'Scale':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))
        elif self.type == 'Power+Scale':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], [None], self.interblob, power_param=self.power_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.interblob[0]], self.inputs_buf, self.outputs, scale_param=self.scale_param))
        elif self.type == 'Power+Reshape+Scale':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], [None], [self.interblob[0]], power_param=self.power_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.interblob[0]], [None], [self.interblob[1]], reshape_param=dict(shape=dict(dim=self.inputs_shape[1]))))
            layers.append(caffe_layer(self.layer_type[2], self.name[2], [self.inputs[0], self.interblob[1]], self.inputs_buf, self.outputs, scale_param=self.scale_param))

        self.setConverted()

        return layers
