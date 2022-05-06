from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Input(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator == 'Placeholder')
        self.setInited()


    def parse(self):
        self.layer_type = 'Input'
        super().__parse__() 

        self.setParsed()


    def convert(self):
        pass
