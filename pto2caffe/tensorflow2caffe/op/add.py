from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Add(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('Add', 'AddV2'))
        self.setInited()


    def parse(self):
        self.layer_type = 'Eltwise'
        super().__parse__()

        self.eltwise_param = dict()
        self.eltwise_param['operation'] = 1
        self.attrs = self.eltwise_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)

        self.setConverted()

        return [layer]
