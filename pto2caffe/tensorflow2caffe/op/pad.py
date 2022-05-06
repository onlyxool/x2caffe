from tensorflow2caffe.op.operator import Operator

class Pad(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator == 'Pad')
        self.setInited()


    def parse(self):
        self.layer_type = 'Pad'
        super().__parse__()

        # Attribute
        self.pad = dict()
        pad_tensor = self.inputs_buf[1]
        self.pad['left'] = pad_tensor[2][0]
        self.pad['right'] = pad_tensor[2][1]
        self.pad['top'] = pad_tensor[1][0]
        self.pad['bottom'] = pad_tensor[1][1]


    def convert(self):
        pass
