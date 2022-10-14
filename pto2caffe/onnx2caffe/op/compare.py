from onnx2caffe.op.operator import Operator


class Compare(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code in ('Equal',))
        self.setInited()


    def parse(self):
        self.type = 'Compare'
        super().__parse__()

        if self.operator_code == 'Equal' and self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0] == self.inputs_buf[1])
        else:
            self.unSupported()


    def convert(self):
        pass
