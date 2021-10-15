import tflite
import logging

from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

PaddingMapping = {
    tflite.Padding.SAME: 'SAME_UPPER',
    tflite.Padding.VALID: 'VALID',
}


class Pad(Operator):

    TypeMapping = {
        tflite.BuiltinOperator.PAD: 'Pad',
        tflite.BuiltinOperator.MIRROR_PAD: 'Pad',
    }


    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.pad = dict()
        self.attrs = self.pad
        self.setInited()


    @property
    def type(self):
        return 'Pad'


    def parse(self):
        logger.debug("Parsing %s...", self.shorty)

        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Option
        pad_tensor = self.inputs_buf[1]
        self.pad['left'] = pad_tensor[2][0]
        self.pad['right'] = pad_tensor[2][1]
        self.pad['top'] = pad_tensor[1][0]
        self.pad['bottom'] = pad_tensor[1][1]


    def convert(self):
        pass
