from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Transpose(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'torch.transpose')
        self.setInited()


    def parse(self):
        self.type = 'Permute'
        super().__parse__()

        order = list()
        for i in range(len(self.inputs_shape[0])):
            order.append(i)

        order[self.attrs['dim0']] = self.attrs['dim1']
        order[self.attrs['dim1']] = self.attrs['dim0']

        # Attributes
        self.permute_param = dict()
        self.permute_param['order'] = order

        self.attrs = self.permute_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]
