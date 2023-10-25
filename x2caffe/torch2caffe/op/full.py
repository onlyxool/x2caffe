import numpy as np
from torch2caffe.op.operator import Operator


class Full(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'full')
        self.setInited()


    def parse(self):
        self.type = 'Full'
        super().__parse__()

        self.saveConstant(self.outputs[0], np.full(shape=self.inputs_buf[0], fill_value=self.inputs_buf[1]))


    def convert(self):
        pass
