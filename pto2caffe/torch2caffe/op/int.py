from torch2caffe.op.operator import Operator


class Int(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'int')
        self.setInited()


    def parse(self):
        self.type = 'int'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], int(self.inputs_buf[0]))
        else:
            self.unSupported()
        

    def convert(self):
        pass
