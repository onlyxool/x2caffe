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
        elif self.inputs_buf[1] is not None and self.inputs_buf[1] == 0:
            self.type = 'Reshape'

            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            self.attrs = self.reshape_param
            self.setParsed()
        else:
            self.unSupported()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]

