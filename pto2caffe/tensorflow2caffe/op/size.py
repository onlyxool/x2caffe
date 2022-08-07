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
            self.model.unsupport.append(self.operator_code)


    def convert(self):
        pass
