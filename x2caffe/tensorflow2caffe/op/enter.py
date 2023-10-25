from tensorflow2caffe.op.operator import Operator


class Enter(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Enter')
        self.setInited()


    def parse(self):
        self.type = 'Enter'
        super().__parse__()

        if self.attrs['is_constant'] or self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0])
        else:
            self.byPassOperator()


    def convert(self):
        pass
