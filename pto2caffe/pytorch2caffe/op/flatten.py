import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class Flatten(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.flatten_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'Flatten'
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        self.flatten_param['axis'] = self.attrs['start_dim']
        self.flatten_param['end_axis'] = self.attrs['end_dim']

        self.attrs = self.flatten_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, flatten_param=self.flatten_param)

        self.setConverted()

        return [layer]
