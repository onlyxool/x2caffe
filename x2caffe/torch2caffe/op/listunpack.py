from torch2caffe.op.operator import Operator


class Listunpack(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'listunpack')
        self.setInited()


    def parse(self):
        self.type = 'Listunpack'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            for index, output in enumerate(self.outputs):
                self.saveConstant(output, self.inputs_buf[0][index])
        else:
            self.unSupported()


    def convert(self):
        pass
