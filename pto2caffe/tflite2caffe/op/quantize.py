import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Quantize(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)

        assert(self.operator in ('QUANTIZE', 'DEQUANTIZE'))

        self.setInited()


    def parse(self):

        self.parseInputOutput()

        if self.inputs_buf[0] is None:
            self.layer_type = 'Reshape'
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            self.setParsed()
        else:
            self.model.tensor[self.outputs[0]] = self.model.tensor[self.inputs[0]]


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
