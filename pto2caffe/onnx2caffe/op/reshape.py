import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Reshape(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.layer_type = 'Constant'
            self.model.input_tensor[self.outputs[0]] = self.inputs_buf[0].reshape(self.outputs_shape[0])
        else:
            self.layer_type = 'Reshape'

            # Attributes
            if 'shape' in self.attrs:
                self.reshape_param = dict(shape=dict(dim=self.attrs['shape']))
            else:
                self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            self.attrs = self.reshape_param
            self.setParsed()


    def convert(self):
        if self.type == 'Reshape':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)
            self.setConverted()
            return [layer]
