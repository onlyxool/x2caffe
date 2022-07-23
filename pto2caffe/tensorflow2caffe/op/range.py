import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

#from util import dtype_tf2np

class Range(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Range')
        self.setInited()


    def parse(self):
        self.layer_type = 'Range'
        super().__parse__()

        start = int(self.inputs_buf[0])
        limit = int(self.inputs_buf[1])
        delta = int(self.inputs_buf[2])

        self.model.constant[self.outputs[0]] = np.arange(start=start, stop=limit, step=delta, dtype=self.op.outputs[0].dtype.as_numpy_dtype())


    def convert(self):
        pass
