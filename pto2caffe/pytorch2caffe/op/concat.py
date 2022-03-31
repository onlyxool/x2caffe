import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class Concat(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.concat_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Concat'


    def parse(self):
        logger.debug('Parsing %s...', self.type)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()
        self.concat_param['axis'] = self.attrs['dim']

        self.attrs = self.concat_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]
