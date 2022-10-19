from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Concat(Operator):
    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'concatenate')
        self.setInited()


    def parse(self):
        super().__parse__()
        self.type = 'Concat'

        self.concat_param = dict()
        self.concat_param['axis'] = self.attrs['axis']

        self.attrs = self.concat_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]
