from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class BatchNorm(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'batch_norm')
        self.setInited()


    def parse(self):
        self.type = 'BatchNorm'
        super().__parse__()

        # Var
        self.var = self.inputBuf_byName('running_var')

        # Mean
        self.mean = self.inputBuf_byName('running_mean')

        # Bias
        self.bias = self.inputBuf_byName('bias')

        # Weight
        self.weight = self.inputBuf_byName('weight')

        # Attributes
        self.batch_norm_param = dict()
        self.batch_norm_param['eps'] = self.inputs_buf[7]
        self.batch_norm_param['use_global_stats'] = self.inputs_buf[5]

        self.scale_param = dict()
        self.scale_param['bias_term'] = self.inputs_buf[8]

        self.attrs = self.batch_norm_param

        self.setParsed()


    def convert(self):
        layer0 = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.mean, self.var, batch_norm_param=self.batch_norm_param)

        layer1 = caffe_layer('Scale', 'Scale'+str(self.index), self.outputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer0, layer1]
