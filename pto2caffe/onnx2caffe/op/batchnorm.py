from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class BatchNorm(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.batch_norm_param = dict()
        self.scale_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'BatchNorm'
        super().__parse__()

        # Weight
        self.weight = self.inputs_buf[1]

        # Bias
        self.bias = self.inputs_buf[2]

        # Mean
        self.mean = self.inputs_buf[3]

        # var
        self.var = self.inputs_buf[4]

        # Attributes
        self.batch_norm_param['eps'] = self.attrs.get('epsilon', 1e-5)
        self.batch_norm_param['use_global_stats'] = True

        self.scale_param['bias_term'] = True

        self.attrs = self.batch_norm_param

        self.setParsed()


    def convert(self):
        layer0 = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.mean, self.var, batch_norm_param=self.batch_norm_param)
        layer1 = caffe_layer('Scale', 'Scale'+str(self.index), self.outputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer0, layer1]
