from onnx2caffe.op.operator import Operator


class Less(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Less')
        self.setInited()


    def parse(self):
        self.type = 'Less'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            import numpy as np
            self.saveConstant(self.outputs[0], np.array(self.inputs_buf[0] < self.inputs_buf[1]))
        else:
            self.unSupported()


    def convert(self):
        pass
