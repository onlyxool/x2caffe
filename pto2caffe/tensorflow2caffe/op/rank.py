import numpy as np

from tensorflow2caffe.op.operator import Operator


class Rank(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Rank')
        self.setInited()


    def parse(self):
        self.layer_type = 'Rank'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.model.constant[self.outputs[0]] = np.array(self.inputs_buf[0].ndim)
        else:
            self.model.unsupport.append(self.operator_code)


    def convert(self):
        pass
