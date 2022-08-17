from tflite2caffe.op.operator import Operator


class Quantize(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code in ('QUANTIZE', 'DEQUANTIZE'))
        self.setInited()


    def parse(self):

        self.parseInputOutput()

        if self.inputs_buf[0] is None:
            self.model.indentity[self.outputs[0]] = self.model.indentity.get(self.inputs[0], self.inputs[0])
            # Handle Legacy Pad for Ignore OP
            if self.op.Inputs(0) in self.model.pad.keys():
                self.model.pad[self.op.Outputs(0)] = self.model.pad[self.op.Inputs(0)]
        else:
            self.model.constant[self.outputs[0]] = self.model.constant[self.inputs[0]]


    def convert(self):
        pass
