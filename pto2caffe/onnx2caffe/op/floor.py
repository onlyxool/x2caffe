from onnx2caffe.op.operator import Operator


class Floor(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Floor')
        self.setInited()


    def parse(self):
        self.layer_type = 'Floor'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            import numpy as np
            self.saveConstant(self.node.output[0], np.floor(self.inputs_buf[0]))
        else:
            self.unSupported()


    def convert(self):
        pass
