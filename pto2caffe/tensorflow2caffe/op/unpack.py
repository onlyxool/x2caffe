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
                self.model.constant[self.outputs[index]] = np.squeeze(outputs_buf[index], axis=self.attrs['axis'])
        else:
            self.layer_type = 'Slice'

            # Attribute axis
            self.slice_param = dict()
            if len(self.inputs_shape) == 4:
                self.slice_param['axis'] = dim_map_nhwc2nchw[self.op.inputs[0].shape.as_list().index(self.attrs['num'])]
            else:
                self.slice_param['axis'] = self.op.inputs[0].shape.as_list().index(self.attrs['num'])

            # Attribute slice_point
            slice_points = np.arange(start=1, stop=self.attrs['num'], step=1, dtype = np.int32).tolist()
            self.slice_param['slice_point'] = slice_points

            self.attrs = self.slice_param

            self.reshapes = list()
            for index, output in enumerate(self.outputs):
                self.reshapes.append('Unpack_' + self.op.name + '_split' + str(index))

            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

            self.setParsed()


    def convert(self):
        layers = list()
        layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.reshapes, slice_param=self.slice_param))
        for index, reshape_name in enumerate(self.reshapes):
            layers.append(caffe_layer('Reshape', reshape_name, [reshape_name], [None], [self.outputs[index]], reshape_param=self.reshape_param))

        self.setConverted()

        return layers
