import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Permute(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.permute_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Permute'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        self.permute_param['order'] = list(self.attrs['perm'])
        self.attrs = self.permute_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]
