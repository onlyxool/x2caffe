from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Select(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'Tensor.select')
        self.setInited()


    def parse(self):
        self.layer_type = 'Slice'
        super().__parse__()

        # Attributes
        self.slice_param = dict()
        self.slice_param['axis'] = self.attrs['dim']
        self.slice_param['slice_point'] = 1

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
