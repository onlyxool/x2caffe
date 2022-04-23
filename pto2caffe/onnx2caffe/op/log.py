import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Log(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.log_param = dict()
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        self.layer_type = 'Log'
        super().__parse__()

        # Attributes
        # Leave all arguments in ExpParameter as default
        # base = -1.0 (base = e)
        # scale = 1.0
        # shift = 0.0

        self.attrs = self.log_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, log_param=self.log_param)

        self.setConverted()

        return [layer]

