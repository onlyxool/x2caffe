from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Concat(Operator):
    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Concat')
        self.setInited()


    def parse(self):
        self.layer_type = 'Concat'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            import numpy as np
            self.saveConstant(self.node.output[0], np.concatenate(self.inputs_buf, axis=self.attrs['axis']))
        else:
            self.concat_param = dict()
            self.concat_param['axis'] = self.attrs['axis']

            self.attrs = self.concat_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]
