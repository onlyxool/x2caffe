import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Concat(Operator):
    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
        return 'Concat'

    def parse(self):
        logger.debug("Parsing %s...", self.shorty)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        self.concat_param = dict()
        self.concat_param['axis'] = self.attrs['axis']
        self.attrs = self.concat_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)
        self.setConverted()
        return [layer]
