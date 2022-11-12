from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw

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
            errorMsg = 'Do not support stride > 1. OP ' + self.op.name + '\'s strides is ' + str(self.inputs_buf[3]) + '\n'
            self.unSupported(errorMsg)
            return

        if self.inputs_shape[0] is not None and len(self.inputs_shape[0]) > 4:
            self.unSupported('Do not support dimitions > 4.')
            return

        axis = self.attrs['axes']
        if self.attrs.get('axes', None) is not None and len(self.attrs['axes']) > 1:
            self.unSupported('Can\'t slice more than one axis')
            return

        start = self.attrs['begin'][0]
        end = self.attrs['end']

        if start == 0:
            slice_point = end
            self.outputs.append(self.name+'_useless')
        elif end == self.inputs_shape[0][axis[0]]:
            slice_point = start
            self.outputs.insert(0, self.name+'_useless')
        else:
            errorMsg = 'Can\'t support begin: ' + str(self.inputs_buf[1]) + ' end: ' + str(self.inputs_buf[2])
            self.unSupported(errorMsg)
            return

        self.slice_param = dict()
        self.slice_param['axis'] = int(axis[0]) if self.layout == 'NCHW' else dim_map_nhwc2nchw[int(axis[0])]
        self.slice_param['slice_point'] = [slice_point]

        self.attrs = self.slice_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
