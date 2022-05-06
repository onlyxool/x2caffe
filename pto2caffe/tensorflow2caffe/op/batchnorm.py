from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class BatchNorm(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator == 'FusedBatchNorm' or self.operator == 'FusedBatchNormV3')
        self.setInited()


    def parse(self):
        self.layer_type = 'BatchNorm'
        super().__parse__()

        # Weight Bias Mean Variance
        for index, input_name in enumerate(self.inputs):
            if 'gamma' in input_name:
                self.weight = self.inputs_buf[index]
            elif 'beta' in input_name:
                self.bias = self.inputs_buf[index]
            elif 'mean' in input_name:
                self.mean = self.inputs_buf[index]
            elif 'variance' in input_name:
                self.var = self.inputs_buf[index]

        # FusedBatchNorm
        if len(self.outputs) > 1:
            self.outputs = self.outputs[0:1]

        # Attribute
        self.batch_norm_param = dict()
        self.batch_norm_param['eps'] = self.attrs.get('epsilon', 1e-5)
        self.batch_norm_param['use_global_stats'] = True

        self.scale_param = dict()
        self.scale_param['bias_term'] = True if hasattr(self, 'bias') else False

        self.attrs = self.batch_norm_param

        self.setParsed()


    def convert(self):
        layer0 = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.mean, self.var, batch_norm_param=self.batch_norm_param)
        layer1 = caffe_layer('Scale', 'Scale'+str(self.index), self.outputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer0, layer1]
