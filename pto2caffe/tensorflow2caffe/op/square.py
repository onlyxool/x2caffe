from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Square(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Square')
        self.setInited()


    def parse(self):
        self.layer_type = 'Power'
        super().__parse__()

        # Attributes
        self.power_param = dict()
        self.power_param['power'] = 2
        self.power_param['scale'] = 1
        self.power_param['shift'] = 0
        
        self.attrs = self.power_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, power_param=self.power_param)

        self.setConverted()

        return [layer]
