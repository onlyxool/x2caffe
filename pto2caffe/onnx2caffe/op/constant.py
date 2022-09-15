from onnx2caffe.op.operator import Operator


class Constant(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        self.type = 'Constant'
        super().__parse__()

        self.saveConstant(self.node.output[0], self.attrs['value'])


    def convert(self):
        pass
