from tflite2caffe.op.operator import Operator


class Quantize(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code in ('QUANTIZE', 'DEQUANTIZE'))
        self.setInited()


    def parse(self):

        self.parseInputOutput()

        if self.inputs_buf[0] is None:
            self.byPassOperator()
        else:
            self.saveConstant(self.outputs[0], self.model.constant[self.inputs[0]])


    def convert(self):
        pass
