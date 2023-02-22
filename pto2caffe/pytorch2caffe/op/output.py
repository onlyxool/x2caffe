from pytorch2caffe.op.operator import Operator


class Output(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'pnnx.Output')
        self.isLegacy = False
        self.setInited()


    def parse(self):
        self.type = 'Output'
        super().__parse__()


    def convert(self):
        pass
