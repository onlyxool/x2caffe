from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Sqrt(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'sqrt')
        self.setInited()


    def parse(self):
        self.type = 'Power'
        super().__parse__()

        self.power_param = dict()
        self.power_param['power'] = 0.5
        self.power_param['scale'] = 1
        self.power_param['shift'] = 0

        self.attrs = self.power_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, power_param=self.power_param)

        self.setConverted()

        return [layer]
