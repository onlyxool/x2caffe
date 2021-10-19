import logging

from tensorflow2caffe.op.operator import Operator

logger = logging.getLogger('TensorFlow2caffe')


class Pad(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.pad = dict()
        self.setInited()


    @property
    def type(self):
        return 'Pad'


    def parse(self):
        logger.debug("Parsing %s...", self.shorty)
        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        # Attribute
        pad_tensor = self.inputs_buf[1]
        self.pad['left'] = pad_tensor[2][0]
        self.pad['right'] = pad_tensor[2][1]
        self.pad['top'] = pad_tensor[1][0]
        self.pad['bottom'] = pad_tensor[1][1]


    def convert(self):
        pass
