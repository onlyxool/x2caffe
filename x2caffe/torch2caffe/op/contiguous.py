from torch2caffe.op.operator import Operator


class Contiguous(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'contiguous')
        self.setInited()


    def parse(self):
        self.type = 'Contiguous'
        super().__parse__()

        self.byPassOperator()


    def convert(self):
        pass
