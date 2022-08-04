from tensorflow2caffe.op.operator import Operator


class Size(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Size')
        self.setInited()


    def parse(self):
        self.layer_type = 'Size'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            import numpy as np
            self.model.constant[self.outputs[0]] = np.array(self.inputs_buf[0].size)
        else:
            import sys
            errorMsg = 'Error: Operator {} [ Size ] does not Support (Input = {}).\n'.format(self.op.name, self.inputs_buf[0])
            sys.exit(errorMsg)


    def convert(self):
        pass
