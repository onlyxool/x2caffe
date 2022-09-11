from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

from onnx2caffe.utility import computePad


class ReduceMean(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'ReduceMean')
        self.setInited()


    def parse(self):
        super().__parse__()

        axes = list()
        input_axes = [i for i in range(len(self.inputs_shape[0]))]
        axes.extend([input_axes[axis] for axis in self.attrs['axes']])

        if len(self.inputs_shape[0]) == 4 and axes == [2, 3]:
            if self.attrs['keepdims']:
                self.layer_type = 'Pooling'
            else:
                self.layer_type = 'Pooling+Reshape'
                self.inter_blob = 'reducemax_split_reshape'

            self.pooling_param = dict()
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
            self.pooling_param['kernel_w'] = self.inputs_shape[0][3] if len(self.inputs_shape[0]) == 4 else 1
            self.pooling_param['stride_h'] = 1
            self.pooling_param['stride_w'] = 1
            self.pooling_param['ceil_mode'] = False

            kernel_size = [self.pooling_param['kernel_h'], self.pooling_param['kernel_w']]

            legacy_pad = self.model.pad.get(self.node.input[0], {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
            padding = computePad('Pooling', self.attrs, self.inputs_shape[0], self.outputs_shape[0], kernel_size, [1, 1], legacy_pad)
            self.pooling_param.update(padding)

            self.attrs = self.pooling_param

            self.setParsed()
        elif input_axes[-len(axes):len(self.inputs_shape[0])] == axes:
            if self.attrs.get('keepdims', True):
                self.layer_type = 'Reduction+Reshape'
                self.inter_blob = 'reducemax_split_reshape'
            else:
                self.layer_type = 'Reduction'

            self.reduction_param = dict()
            self.reduction_param['operation'] = 4
            self.reduction_param['axis'] = axes[0]

            self.attrs = self.reduction_param
            self.setParsed()
        else:
            raise NotImplementedError


    def convert(self):
        layers = list()
        if self.type == 'Pooling':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param))
        elif self.type == 'Pooling+Reshape':
            layers.append(caffe_layer(self.type, 'Pooling'+str(self.index), self.inputs, self.inputs_buf, [self.inter_blob], pooling_param=self.pooling_param))
            layers.append(caffe_layer('Reshape', 'postReshape'+str(self.index), [self.inter_blob], [None], self.outputs, reshape_param=dict(shape=dict(dim=self.outputs_shape[0]))))
        elif self.type == 'Reduction':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reduction_param=self.reduction_param))
        elif self.type == 'Reduction+Reshape':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, [self.inter_blob], reduction_param=self.reduction_param))
            layers.append(caffe_layer('Reshape', 'postReshape'+str(self.index), [self.inter_blob], [None], self.outputs, reshape_param=dict(shape=dict(dim=self.outputs_shape[0]))))


        self.setConverted()

        return layers
