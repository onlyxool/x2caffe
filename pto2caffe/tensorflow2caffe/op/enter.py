from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Enter(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Enter')
        self.setInited()


    def parse(self):
        self.layer_type = 'Enter'
        super().__parse__()

        if self.attrs['is_constant']:
            self.model.constant[self.outputs[0]] = self.inputs_buf[0]
        else:
            import sys
            sys.exit('Error: Operator [ Enter ] does not Support (is_constant = False).\n')


    def convert(self):
        pass
