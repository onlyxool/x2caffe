import numpy as np
from tensorflow2caffe.op.operator import Operator


class Shape(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Shape')
        self.setInited()


    def parse(self):
        self.layer_type = 'Shape'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.model.constant[self.outputs[0]] = np.array(self.inputs_buf[0].shape)
        else:
            if self.op.inputs[0].shape.is_fully_defined():
                self.model.constant[self.outputs[0]] = np.array(self.op.inputs[0].shape.as_list())
            else:
                raise NotImplementedError(self.op.name)


    def convert(self):
        pass 
