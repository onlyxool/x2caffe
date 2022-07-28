from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Sum(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Sum')
        self.setInited()


    def parse(self):
        self.layer_type = 'Sum'
        super().__parse__()

        axes = self.inputs_buf[1]

        # Handle Constant OP
        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            import numpy as np
            self.model.constant[self.outputs[0]] = np.sum(self.inputs_buf[0], axis=tuple(axes), keepdims=self.attrs['keep_dims'])
        else:
            raise NotImplementedError(self.op.name)


    def convert(self):
        pass
