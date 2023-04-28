from torch2caffe.op.operator import Operator


class Detach(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'detach')
        self.setInited()


    def parse(self):
        self.type = 'Detach'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0])
        else:
            self.byPassOperator()
        

    def convert(self):
        pass
