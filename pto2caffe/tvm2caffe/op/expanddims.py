import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator

from util import shape_map_nhwc2nchw

class ExpandDims(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'expand_dims')
        self.setInited()


    def parse(self):
        self.type = 'Reshape'
        super().__parse__()

        if self.inputs_shape[0] == self.outputs_shape[0]:
            self.byPassOperator()
        else:
            dim = self.attrs['axis']

            if self.outputs_shape[0] is not None and all(self.outputs_shape[0]):
                target_shape = list(np.expand_dims(np.random.random(self.inputs_shape[0]), dim).shape)
                target_shape = shape_map_nhwc2nchw(target_shape) if self.layout == 'NHWC' else target_shape
            else:
                errorMsg = 'Can\'t Support Output Shape is ' + str(self.outputs_shape[0])
                self.unSupported(errorMsg)
                return

            self.reshape_param = dict(shape=dict(dim=target_shape))

            self.attrs = self.reshape_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
