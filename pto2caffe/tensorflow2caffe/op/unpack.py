import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw

class Unpack(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Unpack')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            outputs_buf = np.split(self.inputs_buf[0], indices_or_sections=self.attrs['num'], axis=self.attrs['axis'])
            for index, output in enumerate(self.outputs):
                self.saveConstant(self.outputs[index], np.squeeze(outputs_buf[index], axis=self.attrs['axis']))
        else:
            if self.attrs['num'] == 1:
                self.type = 'Reshape'
                self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
                self.attrs = self.reshape_param
                self.setParsed()
            else:
                self.type = 'Slice' + 'Reshape' * len(self.inputs)

                # Attribute axis
                self.slice_param = dict()
                if len(self.inputs_shape) == 4:
                    self.slice_param['axis'] = dim_map_nhwc2nchw[self.op.inputs[0].shape.as_list().index(self.attrs['num'])]
                else:
                    self.slice_param['axis'] = self.op.inputs[0].shape.as_list().index(self.attrs['num'])

                # Attribute slice_point
                slice_points = np.arange(start=1, stop=self.attrs['num'], step=1, dtype = np.int32).tolist()
                self.slice_param['slice_point'] = slice_points

                self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

                self.attrs = self.slice_param

                self.setParsed()


    def convert(self):
        layers = list()
        if self.type.startswith('Slice'):
            layers.append(caffe_layer(self.layer_type[0], self.name, self.inputs, self.inputs_buf, self.interblob, slice_param=self.slice_param))
            for index, input_name in enumerate(self.inputs):
                layers.append(caffe_layer(self.layer_type[index+1], self.name[index+1], self.interblob[index], [None], [self.outputs[index]], reshape_param=self.reshape_param))
        elif self.type == 'Reshape':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param))

        self.setConverted()

        return layers
