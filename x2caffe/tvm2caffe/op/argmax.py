from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Argmax(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'argmax')
        self.setInited()


    def parse(self):
        self.type = 'ArgMax'
        super().__parse__()

        self.argmax_param = dict()
        self.argmax_param['out_max_val'] = True
        self.argmax_param['top_k'] = 1
        self.argmax_param['axis'] = self.attrs['axis'][0]

        self.attrs = self.argmax_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, argmax_param=self.argmax_param)

        self.setConverted()

        return [layer]

