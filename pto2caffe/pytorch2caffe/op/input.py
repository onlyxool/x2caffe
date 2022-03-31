import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class Input(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.isLegacy = False
        self.setInited()


    @property
    def type(self):
        return 'Input'


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        for output in self.outputs:
            self.model.inputs.append(output)


    def convert(self):
        pass
