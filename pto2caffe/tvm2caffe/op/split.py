from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw


class Split(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'split')
        self.setInited()


    def parse(self):
        self.type = 'Slice'
        super().__parse__()

        self.slice_param = dict()
        self.slice_param['axis'] = dim_map_nhwc2nchw[self.attrs['axis']] if self.layout == 'NHWC' else self.attrs['axis']

        # Slice Point
        indices_or_sections = self.attrs['indices_or_sections']
        self.slice_param['slice_point'] = list()
        for i in range(indices_or_sections):
            self.slice_param['slice_point'].append(self.outputs_shape[i][self.slice_param['axis']]*i)
        self.slice_param['slice_point'].remove(0)

        self.attrs = self.slice_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]

