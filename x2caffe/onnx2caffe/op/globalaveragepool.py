import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator
from onnx2caffe.utility import computePad


class GlobalAveragePool(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'GlobalAveragePool')
        self.setInited()


    def parse(self):
        super().__parse__()

        kernel_size = self.inputs_shape[0][-2:] if isinstance(self.inputs_shape[0], list) and len(self.inputs_shape[0]) == 4 else None

        if kernel_size is not None and (kernel_size[0] > 15 or kernel_size[1] > 15):
            self.type = 'Reduction+Reshape+Scale'
            self.reduction_param = dict()
            self.reduction_param['operation'] = 1
            self.reduction_param['axis'] = 2
            self.attrs = self.reduction_param

            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

            self.scale_param = dict()
            self.scale_param['axis'] = 0
            self.scale_param['num_axes'] = 0
            self.weight = 1 / (self.inputs_shape[0][-2] * self.inputs_shape[0][-1])
        else:
            self.type = 'Pooling'

            self.pooling_param = dict()
            self.pooling_param['pool'] = 1
            self.pooling_param['stride_h'] = 1
            self.pooling_param['stride_w'] = 1
            self.pooling_param['ceil_mode'] = False
            self.pooling_param['global_pooling'] = True

            # Padding
            if kernel_size is not None:
                legacy_pad = self.model.pad.get(self.node.input[0], {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
                padding = computePad(self.layer_type, self.attrs, self.inputs_shape[0], self.outputs_shape[0], kernel_size, [1, 1], legacy_pad)
                self.pooling_param.update(padding)

            self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Pooling':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param))
        elif self.type == 'Reduction+Reshape+Scale':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, [self.interblob[0]], reduction_param=self.reduction_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.interblob[0]], [None], [self.interblob[1]], reshape_param=self.reshape_param))
            layers.append(caffe_layer(self.layer_type[2], self.name[2], [self.interblob[1]], [None, self.weight], self.outputs, self.weight, scale_param=self.scale_param))

        self.setConverted()

        return layers
