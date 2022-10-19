from tvm2caffe.op.operator import Operator


class ByPassOperator(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code in ('bypass', 'array'))
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.operator_code == 'bypass':
            self.byPassOperator()
        elif self.operator_code == 'array':
            self.byPassOperator()


    def convert(self):
        pass
