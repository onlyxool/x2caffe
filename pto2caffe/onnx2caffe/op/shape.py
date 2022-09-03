import numpy as np
from onnx2caffe.op.operator import Operator


class Shape(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Shape')
        self.setInited()


    def parse(self):
        self.layer_type = 'Shape'
        super().__parse__()

        if len(self.inputs_shape[0]) > 0:
            self.saveConstant(self.node.output[0], np.array(self.model.shape[self.inputs[0]]))
        else:
            self.unSupported()


    def convert(self):
        pass
