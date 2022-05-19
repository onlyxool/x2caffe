from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Flatten(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'torch.flatten')
        self.setInited()


    def parse(self):
        self.layer_type = 'Flatten'
        super().__parse__()

        self.flatten_param = dict()
        self.flatten_param['axis'] = self.attrs['start_dim']
        self.flatten_param['end_axis'] = self.attrs['end_dim']

        self.attrs = self.flatten_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, flatten_param=self.flatten_param)

        self.setConverted()

        return [layer]
