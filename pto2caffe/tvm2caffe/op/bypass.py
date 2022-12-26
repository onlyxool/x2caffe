from tvm2caffe.op.operator import Operator


class Bypass(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'copy')
        self.setInited()


    def parse(self):
        self.type = 'Bypass'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0])
        else:
            self.byPassOperator()


    def convert(self):
        pass
