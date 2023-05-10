import numpy as np
from torch2caffe.op.operator import Operator


class Meshgrid(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'meshgrid')
        self.setInited()


    def parse(self):
        self.type = 'Meshgrid'
        super().__parse__()

        for input_buf in self.inputs_buf:
            if input_buf is not None:
                constant = True
            else:
                constant = False
                break

        if constant:
            self.saveConstant(self.outputs[0], np.meshgrid(self.inputs_buf[0][0], self.inputs_buf[0][1]))
        else:
            self.byPassOperator()


    def convert(self):
        pass
