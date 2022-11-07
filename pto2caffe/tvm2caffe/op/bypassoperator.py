from tvm2caffe.op.operator import Operator


class ByPassOperator(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code in ('bypass', 'array', 'cast'))
        self.setInited()


    def parse(self):
        super().__parse__()

        self.byPassOperator()


    def convert(self):
        pass
