from tensorflow2caffe.op.operator import Operator


class Merge(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Merge')
        self.setInited()


    def parse(self):
        self.layer_type = 'Merge'
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            # Skip Op
            self.byPassOperator()
        elif self.inputs_buf[0] is not None:
            # Constant Op
            self.saveConstant(self.outputs[0], self.inputs_buf[0])
        else:
            self.unSupported()


    def convert(self):
        pass
