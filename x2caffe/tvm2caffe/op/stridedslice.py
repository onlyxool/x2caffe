import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw, shape_map_nhwc2nchw


class StridedSlice(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'strided_slice')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'Slice'

        if self.inputs_shape[0] == self.outputs_shape[0]:
            self.byPassOperator()
            return

        # Check Stride != 1
        if self.attrs['strides'].count(1) != len(self.attrs['strides']):
            self.unSupported('Do not support stride == ' + str(self.attrs['strides']))
            return

        if self.attrs['axes'] is not None:
            axes = self.attrs['axes']
            for index, axis in enumerate(axes):
                axes[index] = dim_map_nhwc2nchw[axis] if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4 else axis
        else:
            axes = [list(np.array(self.inputs_shape[0]) == np.array(self.outputs_shape[0])).index(False)]

        if self.attrs.get('axes', None) is not None and len(self.attrs['axes']) > 1:
            self.unSupported('Can\'t slice more than one axis')
            return

        if len(self.attrs['begin']) == len(self.inputs_shape[0]):
            start = (shape_map_nhwc2nchw(self.attrs['begin']) if self.layout == 'NHWC' else self.attrs['begin'])[axes[0]]
        else:
            start = self.attrs['begin'][axes[0]]
        if len(self.attrs['end']) == len(self.inputs_shape[0]):
            end = (shape_map_nhwc2nchw(self.attrs['end']) if self.layout == 'NHWC' else self.attrs['end'])[axes[0]]
        else:
            end = self.attrs['end'][axes[0]]


        self.slice_param = dict()
        if start == 0:
            self.slice_param['slice_point'] = [end]
            self.outputs.append('intermediate_' + str(self.index))
        elif end == self.inputs_shape[0][axes[0]]:
            self.slice_param['slice_point'] = [start]
            self.outputs.insert(0, 'intermediate_' + str(self.index))
        else:
            self.slice_param['slice_point'] = [start, end]
            self.outputs.insert(0, 'intermediate_' + str(self.index) + '_0')
            self.outputs.append('intermediate_' + str(self.index) + '_1')

        self.slice_param['axis'] = axes[0]
        for index, point in enumerate(self.slice_param['slice_point']):
            if point == 9223372036854775807:
                self.slice_param['slice_point'][index] = self.inputs_shape[0][self.slice_param['axis']]

        self.attrs = self.slice_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
