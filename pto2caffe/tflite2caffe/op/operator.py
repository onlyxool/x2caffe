import sys
import tflite
import numpy as np
from base_Operator import BaseOperator
from util import shape_map_nhwc2nchw


class Operator(BaseOperator):

    def __init__(self, model, tf_op:tflite.Operator, tf_op_name:str, index:int):
        super().__init__(model, model.graph, index)
        self.op = tf_op
        self.operator_code = tf_op_name
        self.activ_type_code = tflite.ActivationFunctionType.NONE


    def __parseInput__(self):
        for i in range(self.op.InputsLength()):
            if self.op.Inputs(i) >= 0:
                self.inputs.append(self.model.indentity.get(self.op.Inputs(i), self.op.Inputs(i)))
                if not isinstance(self.graph.Tensors(self.op.Inputs(i)).ShapeAsNumpy(), np.ndarray):
                    self.inputs_shape.append([])
                else:
                    self.inputs_shape.append(shape_map_nhwc2nchw(self.graph.Tensors(self.op.Inputs(i)).ShapeAsNumpy().tolist()))
                self.inputs_buf.append(self.model.constant[self.op.Inputs(i)])
            else:
                self.inputs_buf.append(None)


    def __parseOutput__(self):
        for i in range(self.op.OutputsLength()):
            if self.op.Outputs(i) >= 0:
                self.outputs.append(self.op.Outputs(i))
                if not isinstance(self.graph.Tensors(self.op.Outputs(i)).ShapeAsNumpy(), np.ndarray):
                    self.outputs_shape.append([])
                else:
                    self.outputs_shape.append(shape_map_nhwc2nchw(self.graph.Tensors(self.op.Outputs(i)).ShapeAsNumpy().tolist()))


    def parseInputOutput(self):
        self.__parseInput__()
        self.__parseOutput__()


    def byPassOperator(self):
        if len(self.outputs) == 0 or len(self.inputs) == 0:
            sys.exit('Error: Use byPassOperator() after parseInputOutput().')

        self.model.indentity[self.outputs[0]] = self.model.indentity.get(self.inputs[0], self.inputs[0])
        # Handle Legacy Pad for Ignore Op
        if self.op.Inputs(0) in self.model.pad.keys():
            self.model.pad[self.op.Outputs(0)] = self.model.pad[self.op.Inputs(0)]
