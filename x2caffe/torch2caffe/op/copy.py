from torch2caffe.op.operator import Operator


class Copy(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'copy')
        self.setInited()


    def parse(self):
        self.type = 'Copy'
        super().__parse__()

        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.node.output[0], self.inputs_buf[0])
        else:
            self.byPassOperator()


    def convert(self):
        pass
