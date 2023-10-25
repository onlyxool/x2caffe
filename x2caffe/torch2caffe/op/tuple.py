from torch2caffe.op.operator import Operator


class Tuple(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code in ('tupleconstruct', 'tupleunpack'))
        self.setInited()


    def parse(self):
        self.type = self.operator_code
        super().__parse__()

        if self.operator_code == 'tupleconstruct':
            self.byPassOperator()
        elif self.operator_code == 'tupleunpack':
            for index, output in enumerate(self.outputs):
                self.model.indentity[self.outputs[index]] = [self.inputs[index]]


    def convert(self):
        pass
