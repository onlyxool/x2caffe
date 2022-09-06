from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Transpose(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Transpose')
        self.setInited()


    def parse(self):
        self.layer_type = 'Permute'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0].transpose(self.inputs_buf[1].tolist()).numpy())
        else:
            self.permute_param = dict()
            self.permute_param['order'] = self.inputs_buf[1].tolist() #TODO
            self.attrs = self.permute_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]
