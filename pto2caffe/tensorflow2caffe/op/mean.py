from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator
from util import handleLegacyPad


class Mean(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Mean')
        self.setInited()


    def parse(self):
        super().__parse__()

        axis = self.inputs_buf[1]

        if (axis.tolist() == [2, 3] and self.layout == 'NCHW') or (axis.tolist() == [1, 2] and self.layout == 'NHWC'):
            if self.attrs['keep_dims']:
                self.layer_type = 'Pooling'
            else:
                self.layer_type = 'Pooling+Reshape'
                self.reshape = 'Mean_' + self.op.name + '_split'
                self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

            self.pooling_param = dict()
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
            self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
            self.pooling_param['stride'] = 1
            self.pooling_param['ceil_mode'] = False

            # Padding
            legacy_pad = self.model.pad.get(self.op.inputs[0].name, {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
            padding = handleLegacyPad('VALID', self.inputs_shape[0], self.outputs_shape[0], self.pooling_param, legacy_pad, self.type)
            self.pooling_param.update(padding)

            self.attrs = self.pooling_param

            self.setParsed()
        elif (axis.tolist() == 3 and self.layout == 'NHWC') or (axis.tolist() == 1 and self.layout == 'NCHW'):
            self.layer_type = 'Permute+Reduction+Permute'

            self.inter_blob0 = 'prePremute_split' + str(self.index)
            self.inter_blob1 = 'postPremute_split' + str(self.index)

            self.reduction_param = dict()
            self.reduction_param['operation'] = 4
            self.reduction_param['axis'] = 3

            self.attrs = self.reduction_param

            self.setParsed()
        else:
            raise NotImplementedError


    def convert(self):
        layers = list()
        if self.type == 'Pooling':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param))
        elif self.type == 'Pooling+Reshape':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, [self.reshape], pooling_param=self.pooling_param))
            layers.append(caffe_layer('Reshape', self.reshape, [self.reshape], [None], self.outputs, reshape_param=self.reshape_param))
        elif self.type == 'Permute+Reduction+Permute':
            layers.append(caffe_layer('Permute', 'prePermute'+str(sefl.index), self.inputs, self.inputs_buf, [self.inter_blob0], permute_param=dict(order=[0,2,3,1])))
            layers.append(caffe_layer('Reduction', 'Reduction'+str(sefl.index), [self.inter_blob0], self.inputs_buf, [self.inter_blob1], reduction_param=self.reduction_param))
            layers.append(caffe_layer('Permute', 'postPermute'+str(sefl.index), [self.inter_blob1], self.inputs_buf, self.outputs, permute_param=dict(order=[0,3,1,2])))

        self.setConverted()

        return layers
