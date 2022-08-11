import tflite
import numpy as np

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

        self.model.indentity[self.outputs[0]] = self.model.indentity.get(self.inputs[0], self.inputs[0])

        if np.count_nonzero(self.inputs_buf[1]) > 0:
            pad = dict()
            pad['left']   = self.inputs_buf[1][2][0]
            pad['right']  = self.inputs_buf[1][2][1]
            pad['top']    = self.inputs_buf[1][1][0]
            pad['bottom'] = self.inputs_buf[1][1][1]

            self.model.pad[self.inputs[0]] = pad


    def convert(self):
        pass
