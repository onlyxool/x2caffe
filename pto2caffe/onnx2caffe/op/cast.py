from onnx2caffe.op.operator import Operator


class Cast(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Cast')
        self.setInited()


    def parse(self):
        self.type = 'Cast'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.node.output[0], self.inputs_buf[0])
        else:
            self.byPassOperator()


    def convert(self):
        pass
