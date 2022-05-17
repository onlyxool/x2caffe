from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Sum(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Sum')
        self.setInited()


    def parse(self):
        self.layer_type = 'Eltwise'
        super().__parse__()

        # Attributes
        self.eltwise_param = dict()
        self.eltwise_param['operation'] = 1 #Caffe Eltwise SUM
        self.attrs = self.eltwise_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)

        self.setConverted()

        return [layer]
