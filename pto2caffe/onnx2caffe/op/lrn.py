import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class LRN(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.lrn_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'LRN'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()
        self.lrn_param['alpha'] = self.attrs['alpha']
        self.lrn_param['beta'] = self.attrs['beta']
        self.lrn_param['local_size'] = self.attrs['size']

        self.attrs = self.lrn_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, lrn_param=self.lrn_param)

        self.setConverted()

        return [layer]

