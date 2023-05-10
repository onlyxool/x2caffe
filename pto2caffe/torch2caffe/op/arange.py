import numpy as np
from torch2caffe.op.operator import Operator


class Arange(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'arange')
        self.setInited()


    def parse(self):
        self.type = 'Arange'
        super().__parse__()

        start = 0
        stop = self.inputs_buf[0]
        dtype = self.inputs_buf[1]

        self.saveConstant(self.outputs[0], np.arange(start=start, stop=stop, step=None, dtype=dtype))


    def convert(self):
        pass
