from pytorch2caffe.op.operator import Operator


class Input(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'pnnx.Input')
        self.isLegacy = False
        self.setInited()


    def parse(self):
        self.type = 'Input'
        super().__parse__()

        for output in self.outputs:
            self.model.inputs.append(output)


    def convert(self):
        pass
