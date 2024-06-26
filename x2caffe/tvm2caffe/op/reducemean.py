from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class ReduceMean(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'mean')
        self.setInited()


    def parse(self):
        super().__parse__()

        if 'exclude' in self.attrs.keys() and self.attrs['exclude']:
            self.unSupported('Do not support attribute exclude.')
            return

        if (self.layout == 'NCHW' and self.attrs['axis'] == [2, 3]) or (self.layout == 'NHWC' and self.attrs['axis'] == [1, 2]):
            if self.attrs.get('keepdims', False):
                self.type = 'Pooling'
            else:
                self.type = 'Pooling+Reshape'
                self.reshape_param = []

            self.pooling_param = dict()
            self.pooling_param['pool'] = 1
            self.pooling_param['global_pooling'] = True
            self.pooling_param['stride_h'] = 1
            self.pooling_param['stride_w'] = 1
            self.pooling_param['ceil_mode'] = False

            pool_pad = self.model.pad.get(self.relay_inputs[0], [0, 0, 0, 0])
            if pool_pad[0] == pool_pad[2] and pool_pad[1] == pool_pad[3]:
                self.pooling_param['pad_h'] = pool_pad[0]
                self.pooling_param['pad_w'] = pool_pad[1]
            else:
                self.pooling_param['pad_t'] = pool_pad[0]
                self.pooling_param['pad_l'] = pool_pad[1]
                self.pooling_param['pad_b'] = pool_pad[2]
                self.pooling_param['pad_r'] = pool_pad[3]

            self.attrs = self.pooling_param

            self.setParsed()
        elif (self.attrs['axis'] == [-1] or self.attrs['axis'] == [len(self.inputs_shape[0])-1]) and self.layout == 'NCHW':
            if self.attrs.get('keepdims', False):
                self.type = 'Reduction+Reshape'
            else:
                self.type = 'Reduction'

            self.reduction_param = dict()
            self.reduction_param['operation'] = 4
            self.reduction_param['axis'] = self.attrs['axis'][0]
            self.attrs = self.reduction_param
            self.setParsed()
        elif self.attrs['axis'] == 1 and input_axes.index(axes[0]) < input_axes[-1]:
            if self.attrs.get('keepdims', True):
                self.type = 'Permute+Reduction+Permute'
            else:
                self.unSupported('axes:' + str(self.attrs['axis']) + ' input_shape:' + str(self.inputs_shape[0]))
                return

            from copy import deepcopy
            permute0 = deepcopy(input_axes)
            permute1 = deepcopy(input_axes)

            del permute0[axes[0]]
            del permute1[len(self.inputs_shape[0])-1]
            permute1.insert(axes[0], len(self.inputs_shape[0])-1)

            self.permute_param0 = dict()
            self.permute_param1 = dict()
            self.permute_param0['order'] = permute0+[axes[0]]
            self.permute_param1['order'] = permute1


            self.reduction_param = dict()
            self.reduction_param['operation'] = 4
            self.reduction_param['axis'] = axes[0]

            self.attrs = self.reduction_param
            self.setParsed()
        else:
            self.unSupported('axes:' + str(self.attrs['axis']) + ' input_shape:' + str(self.inputs_shape[0]))


    def convert(self):
        layers = list()
        if self.type == 'Pooling':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param))
        elif self.type == 'Pooling+Reshape':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, self.interblob, pooling_param=self.pooling_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], self.interblob, [None], self.outputs, reshape_param=dict(shape=dict(dim=self.outputs_shape[0]))))
        elif self.type == 'Reduction':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reduction_param=self.reduction_param))
        elif self.type == 'Reduction+Reshape':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, self.interblob, reduction_param=self.reduction_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], self.interblob, [None], self.outputs, reshape_param=dict(shape=dict(dim=self.outputs_shape[0]))))
        elif self.type == 'Permute+Reduction+Permute':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, [self.interblob[0]], permute_param=self.permute_param0))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.interblob[0]], self.inputs_buf, [self.interblob[1]], reduction_param=self.reduction_param))
            layers.append(caffe_layer(self.layer_type[2], self.name[2], [self.interblob[1]], self.inputs_buf, self.outputs, permute_param=self.permute_param1))

        self.setConverted()

        return layers
