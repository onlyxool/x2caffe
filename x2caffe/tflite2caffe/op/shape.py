from tflite2caffe.op.operator import Operator


class Shape(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'SHAPE')
        self.setInited()


    def parse(self):
        self.type = 'Shape'
        super().__parse__()

        self.saveConstant(self.outputs[0], self.graph.Tensors(self.op.Inputs(0)).ShapeAsNumpy())


    def convert(self):
        pass
