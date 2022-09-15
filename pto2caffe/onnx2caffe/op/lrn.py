from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class LRN(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'LRN')
        self.setInited()


    def parse(self):
        self.type = 'LRN'
        super().__parse__()

        # Attributes
        self.lrn_param = dict()
        self.lrn_param['alpha'] = self.attrs['alpha']
        self.lrn_param['beta'] = self.attrs['beta']
        self.lrn_param['local_size'] = self.attrs['size']

        self.attrs = self.lrn_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, lrn_param=self.lrn_param)

        self.setConverted()

        return [layer]

