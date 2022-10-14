import numpy as np
from onnx2caffe.op.operator import Operator


class Where(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Where')
        self.setInited()


    def parse(self):
        self.type = 'Where'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None and self.inputs_buf[2] is not None:
            self.saveConstant(self.outputs[0], np.where(self.inputs_buf[0], self.inputs_buf[1], self.inputs_buf[2]))
        else:
            self.unSupported()


    def convert(self):
        pass
