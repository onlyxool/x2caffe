from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Add(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'add')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'Eltwise'
        self.eltwise_param = dict()
        self.eltwise_param['operation'] = 1 # Caffe Eltwise SUM
        self.attrs = self.eltwise_param
        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)

        self.setConverted()

        return [layer]
