from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class BatchNorm(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'nn.BatchNorm2d')
        self.setInited()


    def parse(self):
        self.type = 'BatchNorm'
        super().__parse__()

        # Var
        self.var = self.inputs_buf[self.inputs.index('running_var')]

        # Mean
        self.mean = self.inputs_buf[self.inputs.index('running_mean')]

        # Bias
        self.bias = self.inputs_buf[self.inputs.index('bias')]

        # Weight
        self.weight = self.inputs_buf[self.inputs.index('weight')]

        # Attributes
        self.batch_norm_param = dict()
        self.batch_norm_param['eps'] = self.attrs.get('eps', 1e-5)
        self.batch_norm_param['use_global_stats'] = True

        self.scale_param = dict()
        self.scale_param['bias_term'] = True

        self.attrs = self.batch_norm_param

        self.setParsed()


    def convert(self):
        layer0 = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.mean, self.var, batch_norm_param=self.batch_norm_param)

        layer1 = caffe_layer('Scale', 'Scale'+str(self.index), self.outputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer0, layer1]
