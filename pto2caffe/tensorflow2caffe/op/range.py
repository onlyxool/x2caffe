import numpy as np

from tensorflow2caffe.op.operator import Operator


class Range(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Range')
        self.setInited()


    def parse(self):
        self.layer_type = 'Range'
        super().__parse__()

        if self.inputs_buf[0] is None:
            self.unSupported('Can\'t Support Start = None')
            return
        if self.inputs_buf[1] is None:
            self.unSupported('Can\'t Support limit = None')
            return
        if self.inputs_buf[2] is None:
            self.unSupported('Can\'t Support delta = None')
            return

        start = int(self.inputs_buf[0])
        limit = int(self.inputs_buf[1])
        delta = int(self.inputs_buf[2])

        self.saveConstant(self.outputs[0], np.arange(start=start, stop=limit, step=delta, dtype=self.op.outputs[0].dtype.as_numpy_dtype()))


    def convert(self):
        pass
