from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class BatchNorm(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('FusedBatchNorm', 'FusedBatchNormV3'))
        self.setInited()


    def parse(self):
        self.type = 'BatchNorm+Scale'
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
        else:
            self.weight = self.inputs_buf[1]
            self.bias = self.inputs_buf[2]
            self.mean = self.inputs_buf[3]
            self.var = self.inputs_buf[4]

        # Attribute
        self.batch_norm_param = dict()
        self.batch_norm_param['eps'] = self.attrs.get('epsilon', 1e-5)
        self.batch_norm_param['use_global_stats'] = True

        self.scale_param = dict()
        self.scale_param['bias_term'] = True if hasattr(self, 'bias') else False

        self.attrs = self.batch_norm_param

        self.setParsed()


    def convert(self):
        self.layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, self.outputs[0:1], self.mean, self.var, batch_norm_param=self.batch_norm_param))
        self.layers.append(caffe_layer(self.layer_type[1], self.name[1], self.outputs[0:1], [None, self.weight, self.bias], self.outputs[0:1], self.weight, self.bias, scale_param=self.scale_param))

        self.setConverted()

        return self.layers
