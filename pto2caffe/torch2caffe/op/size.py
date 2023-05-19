from torch2caffe.op.operator import Operator


class Size(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'size')
        self.setInited()


    def parse(self):
        self.type = 'Size'
        super().__parse__()

        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        self.saveConstant(self.outputs[0], self.inputs_shape[0] if self.inputs_buf[1] is None else self.inputs_shape[0][self.inputs_buf[1]])


    def convert(self):
        pass
