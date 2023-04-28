import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Mean(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'mean')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        output_shape = list(np.random.random(self.inputs_shape[0]).mean(axis=tuple(self.inputs_buf[1]), keepdims=self.inputs_buf[2]).shape)

        self.outputs_shape[0] = output_shape
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


    def parse(self):
        super().__parse__()

        if len(self.inputs_shape[0]) == 4 and self.inputs_buf[1] == [2, 3]:
            if self.inputs_buf[2]:
                self.type = 'Pooling'
            else:
                self.type = 'Pooling+Reshape'

            self.pooling_param = dict()
            self.pooling_param['pool'] = 1 
            self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
            self.pooling_param['kernel_w'] = self.inputs_shape[0][3] if len(self.inputs_shape[0]) == 4 else 1
            self.pooling_param['stride_h'] = 1 
            self.pooling_param['stride_w'] = 1 
            self.pooling_param['ceil_mode'] = False

            self.attrs = self.pooling_param
            self.compute_output_shape()
            self.setParsed()
        else:
            self.unSupported('axes:' + str(axes) + ' input_shape:' + str(self.inputs_shape[0]))
            return


    def convert(self):
        layers = list()
        if self.type == 'Pooling':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param))
        elif self.type == 'Pooling+Reshape':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[0]], self.inputs_buf, self.interblob, pooling_param=self.pooling_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], self.interblob, [None], self.outputs, reshape_param=dict(shape=dict(dim=self.outputs_shape[0]))))

        self.setConverted()

        return layers
