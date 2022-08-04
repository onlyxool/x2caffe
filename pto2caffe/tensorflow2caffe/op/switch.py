from tensorflow2caffe.op.operator import Operator


class Switch(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Switch')
        self.setInited()


    def parse(self):
        self.layer_type = 'Switch'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            # Constant Op
            self.model.constant[self.outputs[1]] = self.inputs_buf[0]
            self.model.constant[self.outputs[0]] = self.inputs_buf[0]
        else:
            # Skip Op
            self.model.indentity[self.op.outputs[1].name] = self.model.indentity.get(self.op.inputs[0].name, self.op.inputs[0].name)
            self.model.indentity[self.op.outputs[0].name] = self.model.indentity.get(self.op.inputs[0].name, self.op.inputs[0].name)


    def convert(self):
        pass
