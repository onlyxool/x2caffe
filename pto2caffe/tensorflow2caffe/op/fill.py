from tensorflow2caffe.op.operator import Operator


class Fill(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Fill')
        self.setInited()


    def parse(self):
        self.layer_type = 'Fill'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            import numpy as np
            self.model.constant[self.outputs[0]] = np.empty(shape=self.op.outputs[0].shape.as_list(), dtype=self.op.outputs[0].dtype.as_numpy_dtype())
            self.model.constant[self.outputs[0]].fill(self.inputs_buf[1].item(0))
        else:
            import sys
            errorMsg = 'Error: Operator [ Fill ] does not Support (dims = {} and value = {}).\n'.format(self.inputs_buf[0], self.inputs_buf[1])
            sys.exit(errorMsg)


    def convert(self):
        pass
