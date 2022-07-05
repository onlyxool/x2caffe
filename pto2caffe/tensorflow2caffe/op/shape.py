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

        self.model.constant[self.outputs[0]] = np.array(self.inputs_shape[0])

        self.setParsed()


    def convert(self):
        pass 
