from tensorflow2caffe.op.operator import Operator


class Cast(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Cast')
        self.setInited()


    def parse(self):
        self.layer_type = 'Cast'
        super().__parse__()

        if self.inputs_buf is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0])
        else:
            self.byPassOperator()


    def convert(self):
        pass 
