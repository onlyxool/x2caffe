import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class PReLU(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
        return 'PReLU'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()
        self.slope = self.inputs_buf[1]
        self.prelu_param = dict()
        if self.slope.shape[0] == 1:
            self.prelu_param['channel_shared'] = False
        else:
            self.prelu_param['channel_shared'] = False
        self.attrs = self.prelu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.slope, prelu_param=self.prelu_param)

        self.setConverted()

        return [layer]
