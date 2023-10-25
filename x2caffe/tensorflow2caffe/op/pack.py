from copy import deepcopy

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw


class Pack(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Pack')
        self.setInited()


    def parse(self):
        super().__parse__()

        for input_buf in self.inputs_buf:
            if input_buf is not None:
                constant = True
            else:
                constant = False
                break

        if constant:
            import numpy as np
            self.saveConstant(self.outputs[0], np.stack(self.inputs_buf))
        else:
            self.type = 'Reshape+' * len(self.inputs) + 'Concat'

            axis = self.attrs['axis'] if self.attrs['axis'] > 0 else len(self.inputs_shape[0]) + self.attrs['axis'] + 1
            inter_shape = deepcopy(self.inputs_shape[0])
            inter_shape.insert(axis, 1)
            self.reshape_param = dict(shape=dict(dim=inter_shape))

            # Concat Attribute
            self.concat_param = dict()
            self.concat_param['axis'] = dim_map_nhwc2nchw[self.attrs['axis']] if self.layout == 'NHWC' and self.outputs_shape[0] != self.op.outputs[0].shape.as_list() else self.attrs['axis']

            self.attrs = self.concat_param

            self.setParsed()


    def convert(self):
        layers = list()
        for index, input_name in enumerate(self.inputs):
            layers.append(caffe_layer(self.layer_type[index], self.name[index], [self.inputs[index]], [None], self.interblob[index], reshape_param=self.reshape_param))

        layers.append(caffe_layer(self.layer_type[len(self.inputs)], self.name[len(self.inputs)], self.interblob, self.inputs_buf, self.outputs, concat_param=self.concat_param))

        self.setConverted()

        return layers
