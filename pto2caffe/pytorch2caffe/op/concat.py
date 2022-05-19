from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Concat(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'torch.cat')
        self.setInited()


    def parse(self):
        super().__parse__()
        self.layer_type = 'Concat'

        # Attributes
        self.concat_param = dict()
        self.concat_param['axis'] = self.attrs['dim']

        self.attrs = self.concat_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]
