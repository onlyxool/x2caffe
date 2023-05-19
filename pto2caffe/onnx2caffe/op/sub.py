import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator



class Sub(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Sub')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0] - self.inputs_buf[1])
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 # Caffe Eltwise SUB
            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is not None:
            self.type = 'Bias'

            BiasShape = self.inputs_shape[1]
            CompatibleFlag = self.checkShapeCompatible()
            if CompatibleFlag == 'Squeeze':
                self.type = 'Scale+Reshape+Bias'
                self.reshape_param = dict(shape=dict(dim=self.inputs_shape[1]))
            elif not CompatibleFlag:
                self.unSupported('Inputs incompatible shape for Caffe. ' + str(self.inputs_shape[0]) + ' - ' + str(BiasShape))
                return

            self.bias = self.inputs_buf[1].reshape(self.inputs_shape[1]) * -1

            self.bias_param = dict()
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.bias_param['num_axes'] = len(self.inputs_shape[1])

            self.attrs = self.bias_param
            self.setParsed()
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.type = 'Scale+Bias'

            self.scale_param = dict()
            self.weight = np.ones(self.inputs_shape[1]).astype(np.float32) * -1

            self.scale_param['axis'] = 0
            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.scale_param['bias_term'] = False

            BiasShape = self.inputs_shape[1]
            CompatibleFlag = self.checkShapeCompatible()
            if CompatibleFlag == 'Squeeze':
                self.type = 'Scale+Reshape+Bias'
                self.reshape_param = dict(shape=dict(dim=self.inputs_shape[1]))
            elif not CompatibleFlag:
                self.unSupported('Inputs incompatible shape for Caffe. ' + str(self.inputs_shape[0]) + ' - ' + str(BiasShape))
                return

            self.bias_param = dict()
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.bias_param['num_axes'] = len(self.inputs_shape[1])

            self.attrs = self.bias_param
            self.setParsed()
        elif self.inputs_buf[0] is not None and self.inputs_buf[1] is None:
            self.type = 'Bias+Scale'

            self.inputs.reverse()
            self.inputs_shape.reverse()
            self.inputs_buf.reverse()

            self.weight = np.ones(self.outputs_shape[0]).astype(np.float32) * -1
            self.bias = self.inputs_buf[1] * -1

            self.bias_param = dict()
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.bias_param['num_axes'] = len(self.inputs_shape[1])

            self.scale_param = dict()
            self.scale_param['axis'] = 0
            self.scale_param['num_axes'] = len(self.outputs_shape[0])
            self.scale_param['bias_term'] = False

            self.attrs = self.bias_param
            self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'Bias':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))
        elif self.type == 'Scale+Bias':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], self.inputs_buf, self.interblob, self.weight, scale_param=self.scale_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.interblob[0]], self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))
        elif self.type == 'Scale+Reshape+Bias':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], self.inputs_buf, [self.interblob[0]], self.weight, scale_param=self.scale_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.interblob[0]], [None], [self.interblob[1]], reshape_param=self.reshape_param))
            layers.append(caffe_layer(self.layer_type[2], self.name[2], [self.inputs[0], self.interblob[1]], self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))
        elif self.type == 'Bias+Scale':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, self.interblob, self.bias, bias_param=self.bias_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], self.interblob, [None, self.weight], self.outputs, self.weight, scale_param=self.scale_param))

        self.setConverted()

        return layers
