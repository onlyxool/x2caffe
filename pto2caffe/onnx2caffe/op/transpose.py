from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Permute(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Transpose')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0].transpose(self.attrs['perm']))
        else:
            self.layer_type = 'Permute'
            self.permute_param = dict()
            self.permute_param['order'] = list(self.attrs['perm'])

            self.attrs = self.permute_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]
