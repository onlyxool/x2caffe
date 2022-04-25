import tflite
import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Resize(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)

        assert(self.operator in ('RESIZE_NEAREST_NEIGHBOR', 'RESIZE_BILINEAR'))
        assert(self.op.InputsLength() == 2), "TFLite has only two inputs"
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):

        self.parseInputOutput()

        # Output shape
        output_h = self.outputs_shape[0][2]
        output_w = self.outputs_shape[0][3]

        # Input Shape
        input_h = self.inputs_shape[0][2]
        input_w = self.inputs_shape[0][3]

        # Attributes
        scale_factor = output_h/input_h
        if self.operator == 'RESIZE_NEAREST_NEIGHBOR':
            #if output_h/input_h == output_h//input_h and output_w/input_w == output_w//input_w:
            if scale_factor % 1 == 0:
                # Deconvolution Layer
                self.layer_type = 'Deconvolution'
                self.convolution_param = dict()
                self.convolution_param['bias_term'] = False
                self.convolution_param['num_output'] = self.outputs_shape[0][1]
                self.convolution_param['kernel_h'] = int(scale_factor)
                self.convolution_param['kernel_w'] = int(scale_factor)
                self.convolution_param['stride_h'] = int(scale_factor)
                self.convolution_param['stride_w'] = int(scale_factor)
                self.convolution_param['group'] = self.inputs_shape[0][1]
                self.attrs = self.convolution_param
                # TODO: self.convolution_param['pads']
                self.weight = np.ones((self.outputs_shape[0][1], 1, int(scale_factor), int(scale_factor)), dtype=int)
                self.inputs_buf[1] = self.weight
                self.inputs_shape[1] = self.inputs_buf[1].shape
            else:
                # Upsample Layer
                self.layer_type = 'Upsample'
                self.upsample_param = dict()
                self.upsample_param['scale'] = scale_factor
                self.attrs = self.upsample_param
        elif self.operator == 'RESIZE_BILINEAR':
            # Interp Layer
            self.layer_type = 'Interp'
            op_opt = self.op.BuiltinOptions()
            opt = tflite.ResizeBilinearOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            self.interp_param = dict()
            self.interp_param['align_corners'] = opt.AlignCorners()
            self.interp_param['height'] = self.inputs_buf[1][0]
            self.interp_param['width'] = self.inputs_buf[1][1]

            self.attrs = self.interp_param
            #opt.HalfPixelCenters()

        self.setParsed()


    def convert(self):
        if self.type == 'Deconvolution':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, convolution_param=self.convolution_param)
        elif self.type == 'Upsample':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)
        elif self.type == 'Interp':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, interp_param=self.interp_param)

        self.setConverted()

        return [layer]
