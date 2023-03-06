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
        self.type = 'Pad'
        super().__parse__()

        self.byPassOperator()

        if np.count_nonzero(self.inputs_buf[1]) > 0:
            pad = dict()
            pad['left']   = self.inputs_buf[1][2][0].tolist()
            pad['right']  = self.inputs_buf[1][2][1].tolist()
            pad['top']    = self.inputs_buf[1][1][0].tolist()
            pad['bottom'] = self.inputs_buf[1][1][1].tolist()
            pad['channel'] = self.inputs_buf[1][3].tolist()
            self.model.pad[self.op.Outputs(0)] = pad

            if pad['channel'] != [0, 0]:
                self.model.errorMsg.append('Error: Operator PAD: Caffe do not support pad on Channel. ' + str(self.inputs) + ' -> ' + str(self.outputs))
                self.model.unsupport.append(self.operator_code)


    def convert(self):
        pass
