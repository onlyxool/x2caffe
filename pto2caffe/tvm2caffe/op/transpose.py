from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Permute(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'transpose')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0].transpose(self.attrs['axes'])) 
        else:
            self.type = 'Permute'
            self.permute_param = dict()
            self.permute_param['order'] = self.attrs['axes']

            self.attrs = self.permute_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]
