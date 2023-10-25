import torch

from torch2caffe.op.operator import Operator


class Constant(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'constant')
        self.setInited()


    def parse(self):
        self.type = 'Constant'
        super().__parse__()

        self.saveConstant(self.outputs[0], self.attrs['value'].numpy() if isinstance(self.attrs['value'], torch.Tensor) else self.attrs['value'])


    def convert(self):
        pass
