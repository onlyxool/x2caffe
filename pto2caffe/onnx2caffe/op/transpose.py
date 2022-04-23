from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Permute(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.permute_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'Permute'
        super().__parse__()

        # Attributes 
        self.permute_param['order'] = list(self.attrs['perm'])

        self.attrs = self.permute_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]
