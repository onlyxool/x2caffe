import tflite
import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

from util import handleLegacyPad


class DepthToSpace(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)

        assert(self.operator_code == 'DEPTH_TO_SPACE')
        assert(self.op.InputsLength() == 1)
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):
        self.type = 'Deconvolution'
        super().__parse__()

        op_opt = self.op.BuiltinOptions()
        opt = tflite.DepthToSpaceOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        scale_factor = opt.BlockSize()
        ic = out_channel = self.outputs_shape[0][1]
        in_channel = self.inputs_shape[0][1]

        # Weight
        weight = np.zeros((in_channel, out_channel, scale_factor, scale_factor), dtype='int')
        if scale_factor == 2:
            for i in range(in_channel):
                weight[i+0:i+1,         i:i+1, 0:1, 0:1] = 1 
                weight[i+ic:i+ic+1,     i:i+1, 0:1, 1:2] = 1 
                weight[i+2*ic:i+2*ic+1, i:i+1, 1:2, 0:1] = 1 
                weight[i+3*ic:i+3*ic+1, i:i+1, 1:2, 1:2] = 1 
        elif scale_factor == 3:
            for i in range(in_channel):
                weight[i+0:i+1,         i:i+1, 0:1, 0:1] = 1 
                weight[i+ic:i+ic+1,     i:i+1, 0:1, 1:2] = 1 
                weight[i+2*ic:i+2*ic+1, i:i+1, 0:1, 2:3] = 1 

                weight[i+3*ic:i+3*ic+1, i:i+1, 1:2, 0:1] = 1 
                weight[i+4*ic:i+4*ic+1, i:i+1, 1:2, 1:2] = 1 
                weight[i+5*ic:i+5*ic+1, i:i+1, 1:2, 2:3] = 1 

                weight[i+6*ic:i+7*ic+1, i:i+1, 2:3, 0:1] = 1 
                weight[i+7*ic:i+9*ic+1, i:i+1, 2:3, 1:2] = 1 
                weight[i+8*ic:i+9*ic+1, i:i+1, 2:3, 2:3] = 1
        else:
            self.unSupported('DepthToSpace can\'t support BlockSize:' + str(opt.BlockSize()))
            return

        self.weight = weight

        # Attributes
        self.convolution_param = dict()
        self.convolution_param['num_output'] = out_channel
        self.convolution_param['stride_h'] = scale_factor
        self.convolution_param['stride_w'] = scale_factor
        self.convolution_param['dilation'] = [1, 1]
        self.convolution_param['group'] = 1
        self.convolution_param['kernel_h'] = self.weight.shape[2]
        self.convolution_param['kernel_w'] = self.weight.shape[3]
        self.convolution_param['bias_term'] = False

        # Padding
        legacy_pad = self.model.pad.get(self.op.Inputs(0), {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
        padding = handleLegacyPad('VALID', self.inputs_shape[0], self.outputs_shape[0], self.convolution_param, legacy_pad, self.layer_type)
        self.convolution_param.update(padding)

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
