import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class Output(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.isLegacy = False
        self.setInited()


    @property
    def type(self):
        return 'Output'


    def parse(self):
        self.layer_type = 'Output'
        logger.debug("Parsing %s...", self.type)
        self.parseInput()
        self.parseOutput()
        self.parseAttributes()


    def convert(self):
        pass
