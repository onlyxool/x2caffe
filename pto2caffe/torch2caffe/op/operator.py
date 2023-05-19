from copy import deepcopy

from base_Operator import BaseOperator

from util import isShapeFullyDefined

class Operator(BaseOperator):

    def __init__(self, model, graph, node, index):
        super().__init__(model, graph, index)
        self.node = node
        self.inputs = deepcopy(self.node.inputs)
        self.outputs = deepcopy(self.node.outputs)
        self.operator_code = node.kind 


    def __parseInput__(self):
        inputs = list()
        for index, input_name in enumerate(self.inputs):
            inputs = inputs + self.model.indentity.get(input_name, [input_name])
        self.inputs = inputs

        for index, input_name in enumerate(self.inputs):
            self.inputs_shape.append(self.model.tensor_shape.get(input_name, None))
            self.inputs_buf.append(self.model.constant.get(input_name, None))


    def __parseOutput__(self):
        for index, output_name in enumerate(self.outputs):
            self.outputs_shape.append(deepcopy(self.inputs_shape[0]) if len(self.inputs) >= 1 else None)
            self.outputs_buf.append(self.node.outputs_buf[index])
            self.model.tensor_shape[output_name] = self.outputs_shape[index]


    def __parseAttributes__(self):
        self.attrs = self.node.attr


    def __parse__(self):
        self.__parseInput__()
        self.__parseOutput__()
        self.__parseAttributes__()


    def post_forward(self, outputs):
        for index, output in enumerate(outputs):
            self.model.variable[self.outputs[index]] = output
            self.model.tensor_shape[self.outputs[index]] = self.outputs_shape[index] = list(output.shape)


    def isInputShapeFullyDefined(self, index):
        if index > 0:
            return isShapeFullyDefined(self.inputs_shape[index])
        else:
            for input_shape in self.inputs_shape[:index]:
                if isShapeFullyDefined(input_shape):
                    continue
                else:
                    return False

            return True


    def byPassOperator(self, input_index=None):
        self.type = 'ByPassOperator'
        if len(self.outputs) == 0 or len(self.inputs) == 0:
            import sys 
            sys.exit('Error: Use byPassOperator() after __parse__().')

        self.model.indentity[self.outputs[0]] = list()
        if input_index is None:
            for index, input_name in enumerate(self.inputs):
                if self.inputs_buf[index] is None:
                    self.model.indentity[self.outputs[0]].append(self.model.indentity.get(input_name, input_name))
        else:
            self.model.indentity[self.outputs[0]] = self.model.indentity.get(self.inputs[input_index], [self.inputs[input_index]])

        # Handle Legacy Pad for Ignore Op
        if self.node.inputs[0] in self.model.pad.keys():
            self.model.pad[self.node.outputs[0]] = self.model.pad[self.node.inputs[0]]


    def unSupported(self, errorMsg=None):
        if errorMsg is not None:
            self.model.errorMsg.append('Error: Op ' + self.node.name + ' (' + self.operator_code + '): ' + errorMsg)
        self.model.unsupport.append(self.operator_code)
