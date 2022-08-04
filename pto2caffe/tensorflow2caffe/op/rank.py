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
            import sys
            errorMsg = 'Error: Operator {} [ Rank ] does not Support (Input = {}).\n'.format(self.op.name, self.inputs_buf[0])
            sys.exit(errorMsg)


    def convert(self):
        pass
