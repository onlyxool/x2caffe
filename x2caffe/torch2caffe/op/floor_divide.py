from torch2caffe.op.operator import Operator


class Floor_divide(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'floor_divide')
        self.setInited()


    def parse(self):
        self.type = 'Floor_divide'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0] / self.inputs_buf[1])
        else:
            self.unSupported()
        

    def convert(self):
        pass
