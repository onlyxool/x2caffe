import numpy as np

from tflite2caffe.op.operator import Operator


class Pack(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'PACK')
        self.setInited()


    def parse(self):
        self.type = 'Pack'

        self.parseInputOutput()

        for input_buf in self.inputs_buf:
            if input_buf is not None:
                constant = True
            else:
                constant = False
                break

        if constant:
            self.saveConstant(self.outputs[0], np.stack(self.inputs_buf))
        else:
            raise NotImplementedError


    def convert(self):
        pass
