from onnx2caffe.op.operator import Operator


class Constant(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code in ('Constant', 'ConstantOfShape'))
        self.setInited()


    def parse(self):
        self.type = 'Constant'
        super().__parse__()

        if self.operator_code == 'Constant':
            self.saveConstant(self.outputs[0], self.attrs['value'])
        elif self.operator_code == 'ConstantOfShape' and self.inputs_buf[0] is not None:
            import numpy as np
            self.saveConstant(self.outputs[0], np.ones(self.inputs_buf[0], dtype=self.attrs['value'].dtype) * self.attrs['value'])
        else:
            self.unSupported()


    def convert(self):
        pass
