import numpy as np
from onnx2caffe.op.operator import Operator


class NonZero(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'NonZero')
        self.setInited()


    def parse(self):
        self.type = 'NonZero'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            print(self.inputs_buf[0])
            print(np.array(np.nonzero(self.inputs_buf[0])), type(np.array(np.nonzero(self.inputs_buf[0]))))
            self.saveConstant(self.outputs[0], np.nonzero(self.inputs_buf[0])[0])
        else:
            self.unSupported()


    def convert(self):
        pass
