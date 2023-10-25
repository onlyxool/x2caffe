from torch2caffe.op.operator import Operator


class Listconstruct(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'listconstruct')
        self.setInited()


    def parse(self):
        self.type = 'Listconstruct'
        super().__parse__()

        for input_buf in self.inputs_buf:
            if input_buf is not None:
                constant = True
            else:
                constant = False
                break

        if constant:
            self.saveConstant(self.outputs[0], [self.model.constant[input_name] for input_name in self.inputs])
        else:
            self.byPassOperator()


    def convert(self):
        pass
