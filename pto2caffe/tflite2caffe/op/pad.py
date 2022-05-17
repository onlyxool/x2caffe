import tflite

from tflite2caffe.op.operator import Operator


class Pad(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code in ('PAD', 'MIRROR_PAD'))
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):
        self.layer_type = 'Pad'

        self.parseInputOutput()

        # Attributes
        pad_tensor = self.inputs_buf[1]
        self.pad = dict()
        self.pad['left'] = pad_tensor[2][0]
        self.pad['right'] = pad_tensor[2][1]
        self.pad['top'] = pad_tensor[1][0]
        self.pad['bottom'] = pad_tensor[1][1]

        self.attrs = self.pad

    def convert(self):
        pass
