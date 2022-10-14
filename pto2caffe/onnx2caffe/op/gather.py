from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Gather(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Gather')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None and self.inputs_buf[1].size > 0:
            import numpy as np
            self.saveConstant(self.node.output[0], np.take(self.inputs_buf[0], indices=self.inputs_buf[1], axis=self.attrs.get('axis', 0)))
        elif self.inputs_buf[1] is not None and self.inputs_buf[1].size == 1: #TODO
            self.type = 'Slice'

            self.slice_param = dict()
            self.slice_param['axis'] = self.attrs['axis']
            self.slice_param['slice_point'] = 1#self.inputs_shape[0][self.attrs['axis']]
            self.attrs = self.slice_param
            self.setParsed()
        else:
            self.unSupported()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]

