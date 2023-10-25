import numpy as np

from torch2caffe.op.operator import Operator


class Numtotensor(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'numtotensor')
        self.setInited()


    def parse(self):
        self.type = 'Numtotensor'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], np.array(self.inputs_buf[0]))
        else:
            self.unSupported()
        

    def convert(self):
        pass
