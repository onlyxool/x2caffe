import copy
import tflite
import logging
import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')


def trim_one(scale_shape):
    if scale_shape == [] or scale_shape is None:
        return scale_shape

    # Remove 1 from head
    while True:
        if scale_shape[0] == 1:
            scale_shape.remove(1)
        else:
            break

    # Remove 1 from tail
    while True:
        if scale_shape[-1] == 1:
            scale_shape.pop()
        else:
            break

    return scale_shape


def compute_scale_axis(bottom_shape, scale_shape):
    '''
    The first axis of bottom[0] (the first input Blob) along which to apply
    bottom[1] (the second input Blob).  May be negative to index from the end
    (e.g., -1 for the last axis).

    For example, if bottom[0] is 4D with shape 100x3x40x60, the output
    top[0] will have the same shape, and bottom[1] may have any of the
    following shapes (for the given value of axis):
       (axis == 0 == -4) 100; 100x3; 100x3x40; 100x3x40x60
       (axis == 1 == -3)          3;     3x40;     3x40x60
       (axis == 2 == -2)                   40;       40x60
       (axis == 3 == -1)                                60
    Furthermore, bottom[1] may have the empty shape (regardless of the value of
    "axis") -- a scalar multiplier.
    '''
    if scale_shape == []:
        return 0

    shapeA = np.array(bottom_shape)
    shapeB = np.array(scale_shape)

    for i in range(len(shapeA)):
        shape_map = (shapeA[i:(len(shapeB)+i)] == shapeB)
        if isinstance(shape_map, list) and shape_map.count(True) == len(shapeB):
            return i

    return None

class Binary(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.ADD: 'Add',
        tflite.BuiltinOperator.MUL: 'Mul',
        tflite.BuiltinOperator.DIV: 'Div',
        tflite.BuiltinOperator.SUB: 'Sub',
        tflite.BuiltinOperator.POW: 'Pow',
    }

    OptionMapping = {
        tflite.BuiltinOperator.ADD: tflite.AddOptions,
        tflite.BuiltinOperator.MUL: tflite.MulOptions,
        tflite.BuiltinOperator.DIV: tflite.DivOptions,
        tflite.BuiltinOperator.SUB: tflite.SubOptions,
    }

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()

    @property
    def type(self):
        if hasattr(self, 'eltwise_param'):
            return 'Eltwise'
        elif hasattr(self, 'scale_param'):
            return 'Scale'
        elif hasattr(self, 'axpy_param'):
            return 'Axpy'
        else:
            return self.op_code
#        if self.op_code == tflite.BuiltinOperator.ADD:
#            if hasattr(self, 'eltwise_param'):
#                return 'Eltwise'
#            elif hasattr(self, 'scale_param'):
#                return 'Scale'
#        elif self.op_code == tflite.BuiltinOperator.SUB:
#            return 'TODO:SUB'
#        elif self.op_code == tflite.BuiltinOperator.MUL:
#            return 'Scale'
#        elif self.op_code == tflite.BuiltinOperator.DIV:
#            return 'TODO:DIV'
#        elif self.op_code == tflite.BuiltinOperator.POW:
#            return 'TODO:POW'
#        else:
#            raise NotImplementedError

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code in self.TypeMapping)
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Option
        op_opt = self.op.BuiltinOptions()
        if self.op_code == tflite.BuiltinOperator.ADD:
            opt = tflite.AddOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            if self.inputs_buf[1] is not None:
                self.scale_param = dict()
                self.weight = np.ones(self.inputs_shape[1], dtype=int, order='C')
                self.bias = self.inputs_buf[1]
                if self.inputs_shape[1] != []:
                    self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])
                self.scale_param['bias_term'] = True
                self.attrs = self.scale_param
            else:
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 1
                self.attrs = self.eltwise_param
        elif self.op_code == tflite.BuiltinOperator.SUB:
            opt = tflite.SubOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            raise NotImplementedError('SubOptions')
        elif self.op_code == tflite.BuiltinOperator.MUL:
            opt = tflite.MulOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)

            if self.inputs_shape[0] != self.inputs_shape[1] or self.inputs_buf[1] is not None:
                self.scale_param = dict()

                org_shape = copy.deepcopy(self.inputs_shape[1])
                trim = trim_one(org_shape)
                if trim != self.inputs_shape[1]:
                    self.pre.append('Reshape')
                    self.inputs_shape[1] = trim
                    if self.inputs_buf[1] is not None:
                        self.inputs_buf[1] = self.inputs_buf[1].reshape(tuple(trim))

                axis = compute_scale_axis(self.inputs_shape[0], trim)
                if axis is not None:
                    self.scale_param['axis'] = axis

                self.weight = self.inputs_buf[1]
                self.scale_param['bias_term'] = False
                self.bias = None
                self.attrs = self.scale_param
            else:
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 0
                self.attrs = self.eltwise_param
        elif self.op_code == tflite.BuiltinOperator.DIV:
            opt = tflite.DivOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            raise NotImplementedError('DivOptions')
        elif self.op_code == tflite.BuiltinOperator.POW:
            opt = tflite.PowOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            raise NotImplementedError('PowOptions')

        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()


    def convert(self):
        if hasattr(self, 'eltwise_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif hasattr(self, 'scale_param'):
            for pre_op in self.pre:
                if pre_op == 'Reshape':
                    reshape_param = dict(shape=dict(dim=self.inputs_shape[1]))
                    pre_layer = caffe_layer('Reshape', 'Reshape'+str(self.index), [self.inputs[1]], [None], ['reshape'+str(self.index)], reshape_param=reshape_param)
                    self.inputs[1] = 'reshape' + str(self.index)

            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)
            if len(self.pre) > 0:
                self.setConverted()
                return [pre_layer, layer]
        elif hasattr(self, 'axpy_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, axpy_param=self.axpy_param)

#        if self.op_code == tflite.BuiltinOperator.ADD:
#            if hasattr(self, 'eltwise_param'):
#                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
#            elif hasattr(self, 'scale_param'):
#                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)
#        elif self.op_code == tflite.BuiltinOperator.MUL:
#            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
