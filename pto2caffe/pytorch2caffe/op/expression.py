import sys
import copy
import numpy as np

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

#from util import trim_one
#from util import compute_scale_axis


#def trim_one(inputs_shape):
#    reshape = copy.deepcopy(inputs_shape)
#    reshape_flag = []
#    for i, input_shape in enumerate(inputs_shape):
#        if input_shape == [] or input_shape is None:
#            continue 
#
#        # Remove 1 from head
#        while True:
#            if input_shape[0] == 1:
#                input_shape.remove(1)
#            else:
#                break
#    
#        # Remove 1 from tail
#        while True:
#            if input_shape[-1] == 1:
#                input_shape.pop()
#            else:
#                break
#        reshape_flag.append(not(len(reshape[i]) == len(input_shape)))
#
#    return reshape


def trim_one(input_shape):
    ret = False
    if input_shape == [] or input_shape is None:
        return ret

    #Remove 1 from head
    while True:
        if input_shape[0] == 1:
            input_shape.remove(1)
            ret = True
        else:
            break

    # Remove 1 from tail
    while True:
        if input_shape[-1] == 1:
            input_shape.pop()
            ret = True
        else:
            break

    return ret

def switch(iterl):
    temp = iterl[0]
    iterl[0] = iterl[1]
    iterl[1] = temp


def need_switch(inputs_shape):
    if len(inputs_shape[1]) > len(inputs_shape[0]):
        return True
    elif len(inputs_shape[1]) == len(inputs_shape[0]):
        if inputs_shape[0].count(1) > inputs_shape[1].count(1):
            return True
        else:
            return False
    else:
         return False

#def trim_one(inputs_shape):
#    reshape = copy.deepcopy(inputs_shape)
#    for i, input_shape in enumerate(reshape):
#        if input_shape == [] or input_shape is None:
#            continue 
#        # Remove 1 from tail
#        while True:
#            if input_shape[-1] == 1:
#                input_shape.pop()
#            else:
#                break
#    return reshape

#def compute_scale_axis(bottom_shape, scale_shape):
#    ''' 
#    The first axis of bottom[0] (the first input Blob) along which to apply
#    bottom[1] (the second input Blob).  May be negative to index from the end
#    (e.g., -1 for the last axis).
#
#    For example, if bottom[0] is 4D with shape 100x3x40x60, the output
#    top[0] will have the same shape, and bottom[1] may have any of the
#    following shapes (for the given value of axis):
#       (axis == 0 == -4) 100; 100x3; 100x3x40; 100x3x40x60
#       (axis == 1 == -3)          3;     3x40;     3x40x60
#       (axis == 2 == -2)                   40;       40x60
#       (axis == 3 == -1)                                60
#    Furthermore, bottom[1] may have the empty shape (regardless of the value of
#    "axis") -- a scalar multiplier.
#    '''
#    if scale_shape == []: 
#        return 0
#
#    shapeA = np.array(bottom_shape)
#    shapeB = np.array(scale_shape)
#
#    print(shapeA, shapeB, '------------')
#    for i in range(len(shapeA)):
#        shape_map = (shapeA[i:(len(shapeB)+i)] == shapeB)
#        print(shape_map, type(shape_map), type(np.array), len(shapeB))
##        if (type(shape_map) == type(np.ndarray)) and shape_map.count(True) == len(shapeB):
#        if isinstance(shape_map, np.ndarray) and list(shape_map).count(True) == len(shapeB):
#            return i
#
#    return None

def compute_scale_axis(bottom_shape, scale_shape):
    if not isinstance(bottom_shape, list) and not isinstance(bottom_shape, tuple):
        raise NotImplementedError

    if len(bottom_shape) > 4 or len(bottom_shape) < 2:
        raise NotImplementedError

    bottom_map = \
    [[bottom_shape[:1], bottom_shape[:2], bottom_shape[:3], bottom_shape[:4]],
                       [bottom_shape[1:2], bottom_shape[1:3], bottom_shape[1:4]],
                                          [bottom_shape[2:3], bottom_shape[2:4]],
                                                             [bottom_shape[3:4]]]
    for no, line in enumerate(bottom_map):
        for colum in line:
            if colum == scale_shape:
                return no


#def compute_scale_axis(bottom_shape, scale_shape):
#    if scale_shape == [] or scale_shape == ():
#        return 0
#
#    bottom_shape_np = np.array(bottom_shape)
#    scale_shape_np = np.array(scale_shape)
#
#    shape_flag = []
#    for i in range(bottom_shape_np.size - scale_shape_np.size + 1):
#        shape_map = list(bottom_shape_np[i:(scale_shape_np.size+i)] == scale_shape_np)
#
#        if i > 0:
#            shape_map.insert(i, False)
#
#        for j in range(bottom_shape_np.size - len(shape_map)):
#            shape_map.append(False)
#
#        if shape_map.count(True) > shape_flag.count(True):
#            shape_flag = shape_map
#
#    return shape_flag


class Expression(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.setInited()


    def parse(self):
        super().__parse__()

        # Attributes
        expr = self.attrs['expr']
        if expr.find('add') != -1:
            if self.inputs_shape[0] == self.inputs_shape[1]:
                # Eltwise Layer
                self.layer_type = 'Eltwise'
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 1 # Caffe Eltwise SUM
                self.attrs = self.eltwise_param
            else:
                raise NotImplementedError
        elif expr.find('mul') != -1:
            if self.inputs_shape[0] == self.inputs_shape[1]:
                # Eltwise Layer
                self.layer_type = 'Eltwise'
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 0 # Caffe Eltwise PROD
                self.attrs = self.eltwise_param
            else:
                # Scale Layer
                self.layer_type = 'Scale'
                self.scale_param = dict()

                self.scale_param['bias_term'] = False
                self.weight = None
                self.bias = None

                if need_switch(self.inputs_shape):
                    switch(self.inputs)
                    switch(self.inputs_shape)

                if trim_one(self.inputs_shape[1]):
                    self.pre = ['reshape']

                self.scale_param['axis'] = compute_scale_axis(self.inputs_shape[0], self.inputs_shape[1])
                self.scale_param['num_axes'] = len(self.inputs_shape[1])
                        
                self.attrs = self.scale_param

        self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
            self.setConverted()
            return [layer]
        elif self.type == 'Scale':
            if self.pre == ['reshape']:
                reshape_param = dict(shape=dict(dim=self.inputs_shape[1]))
                pre_layer = caffe_layer('Reshape', 'Reshape'+str(self.index), [self.inputs[1]], [None], ['reshape'+str(self.index)], reshape_param=reshape_param)
                self.inputs[1] = 'reshape' + str(self.index)
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)
            self.setConverted()
            return [pre_layer, layer]
