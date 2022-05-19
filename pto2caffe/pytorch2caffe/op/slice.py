from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Slice(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'Tensor.slice')
        self.setInited()


    @property
    def type(self):
        return 'Slice'


    def parse(self):
        self.layer_type = 'Slice'
        super().__parse__()

        self.slice_param = dict()

        self.setParsed()


    def convert(self):
        pass
#        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, slice_param=self.slice_param)
#        self.setConverted()
#
#        return [layer]
