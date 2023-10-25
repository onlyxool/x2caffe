from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class HardSigmoid(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'HardSigmoid')
        self.setInited()


    def parse(self):
        self.type = 'HardSigmoid'
        super().__parse__()

        if round(self.attrs.get('alpha', 0.2), 3) != 0.2 or round(self.attrs.get('beta', 0.5), 3) != 0.5:
            self.unSupported('Can\'t Support alpha:' + str(self.attrs['alpha']) + ' beta:' + str(self.attrs['beta']))
            return

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs)

        self.setConverted()

        return [layer]
