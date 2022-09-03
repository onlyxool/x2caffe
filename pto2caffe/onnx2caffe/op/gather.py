import torch
from onnx2caffe.op.operator import Operator


class Gather(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Gather')
        self.setInited()


    def parse(self):
        self.layer_type = 'Gather'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[0].size > 0:
            output = torch.gather(torch.tensor(self.inputs_buf[0]), dim=self.attrs['axis'], index=torch.tensor(self.inputs_buf[1]), sparse_grad=False)
            self.saveConstant(self.node.output[0], output.numpy())
        else:
            self.unSupported()


    def convert(self):
        pass
