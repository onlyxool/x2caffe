from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Permute(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'torch.permute')
        self.setInited()


    def parse(self):
        self.layer_type = 'Permute'
        super().__parse__()

        # Attributes
        self.permute_param = dict()
        self.permute_param['order'] = self.attrs['dims']
        self.attrs = self.permute_param
        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]
