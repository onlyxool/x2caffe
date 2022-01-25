import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Unsqueeze(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
        if hasattr(self, 'reshape_param'):
            return 'Reshape'
        else:
            return 'Unsqueeze'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()

        if self.inputs_buf[0] is not None:
            self.model.input_tensor[self.outputs[0]] = self.inputs_buf[0].reshape(self.outputs_shape[0])
        else:
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            self.attrs = self.reshape_param
            self.setParsed()


    def convert(self):
        if hasattr(self, 'reshape_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)
            self.setConverted()
            return [layer]
