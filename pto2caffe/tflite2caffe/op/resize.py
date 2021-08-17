import tflite
import logging
import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class Resize(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: 'Resize',
        tflite.BuiltinOperator.RESIZE_BILINEAR: 'Resize',
    }

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)

        self.setInited()

    @property
    def type(self):
        if self.op_code == tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            if hasattr(self, 'convolution_param'):
                return 'Deconvolution'
            elif hasattr(self, 'upsample_param'):
                return 'Upsample'
            else:
                return 'nearest'
        elif self.op_code == tflite.BuiltinOperator.RESIZE_BILINEAR:
            return 'Interp'
        else:
            raise NotImplementedError

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code in self.TypeMapping)
        assert(self.op.InputsLength() == 2), "TFLite has only two inputs"
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Output shape
        output_h = self.outputs_shape[0][2]
        output_w = self.outputs_shape[0][3]

        # Input Shape
        input_h = self.inputs_shape[0][2]
        input_w = self.inputs_shape[0][3]

        # Option
        scale_factor = int(output_h/input_h)
        if self.op_code == tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            if output_h/input_h == output_h//input_h and output_w/input_w == output_w//input_w:
                self.convolution_param = dict()
                self.convolution_param['bias_term'] = False
                self.convolution_param['num_output'] = self.outputs_shape[0][1]
                self.convolution_param['kernel_h'] = scale_factor
                self.convolution_param['kernel_w'] = scale_factor
                self.convolution_param['stride_h'] = scale_factor
                self.convolution_param['stride_w'] = scale_factor
                self.convolution_param['group'] = self.inputs_shape[0][1]
                self.attrs = self.convolution_param
                # TODO: self.convolution_param['pads']
                self.weight = np.ones((self.outputs_shape[0][1], 1, scale_factor, scale_factor), dtype=int)
                self.inputs_buf[1] = self.weight
                self.inputs_shape[1] = self.inputs_buf[1].shape
            else:
                self.upsample_param = dict()
                self.upsample_param['scale'] = scale_factor
                self.attrs = self.upsample_param
        elif self.op_code == tflite.BuiltinOperator.RESIZE_BILINEAR:
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
        if self.op_code == tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            if hasattr(self, 'convolution_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
            elif hasattr(self, 'upsample_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)
            else:
                raise NotImplementedError
        elif self.op_code == tflite.BuiltinOperator.RESIZE_BILINEAR:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, interp_param=self.interp_param)
        else:
            raise NotImplementedError

        self.setConverted()
        return [layer]

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass
