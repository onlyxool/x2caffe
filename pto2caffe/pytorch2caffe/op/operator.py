from base import Base

class Operator(Base):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, None, index)
        self.operator_code = type_code
        self.layer_type = str()
        self.pnnx = pnnx

        self.inputs = []
        self.inputs_shape = []
        self.inputs_buf = []

        self.outputs = []
        self.outputs_shape = []

        self.pre = []
        self.post = []

        self.attrs = dict()


    @property
    def type(self):
        return self.layer_type if self.layer_type is not None else self.operator_code


    @property
    def name(self):
        return self.pnnx.get_ops_name(self.index)


    @property
    def shorty(self):
        return '[%s](%s)' % (self.name, self.type)


    def str(self):
        return '[' + self.name + ']  (' + self.type + ')'


    @property
    def attrs2str(self):
        attrstr = ''
        for key, value in self.attrs.items():
            attrstr = attrstr + '    ' + str(key) + ': ' + str(value) + '\n'
        return attrstr


    def __str__(self):
        inames = str([t for t in self.inputs])
        onames = str([t for t in self.outputs])
        return '\n%s\n%s    %s -> %s' % (self.shorty, self.attrs2str, inames, onames)


    def __byPassLegacy__(self):
        for legacy in self.model.legacys:
            if legacy.operator_code == 'F.pad':
                for i, input_name in enumerate(self.inputs):
                    if legacy.outputs[0] == self.inputs[i]:
                        self.inputs[i] = legacy.inputs[0]
                        self.inputs_shape[i] = legacy.inputs_shape[0]
                        if legacy.attrs['pad'] == [0,0,0,0]:
                            return True
                        else:
                            raise NotImplementedError


    def __parseInput__(self):
        self.inputs = self.pnnx.get_ops_inputs(self.index)
        for i, input_name in enumerate(self.inputs):
            self.inputs_shape.append(self.pnnx.get_ops_input_shape(self.index, i, input_name))
            tensor = self.pnnx.get_ops_attr(self.index, input_name)
            self.inputs_buf.append(tensor.reshape(self.inputs_shape[i]) if tensor is not None else None)


    def __parseOutput__(self):
        self.outputs = self.pnnx.get_ops_outputs(self.index)
        for i, output_name in enumerate(self.outputs):
            self.outputs_shape.append(self.pnnx.get_ops_output_shape(self.index, i))


    def __parseAttributes__(self):
        params = self.pnnx.get_ops_param(self.index).split('|')
        for param in params:
            if param == '':
                continue
            type_str = param.split('@')[-1]
            key = param.split('@')[0].split('=')[0]
            value_str = param.split('@')[0].split('=')[1]

            if type_str == 'None':
                self.attrs[key] = None
            elif type_str == 'bool':
                self.attrs[key] = True if value_str == 'True' else False
            elif type_str == 'int':
                self.attrs[key] = int(value_str)
            elif type_str == 'float':
                self.attrs[key] = float(value_str)
            elif type_str == 'string':
                self.attrs[key] = value_str
            elif type_str == '[int]':
                self.attrs[key] = [] 
                value_str_list = value_str.split(',')
                for value_str_item in value_str_list:
                    self.attrs[key].append(int(value_str_item))
            elif type_str == '[float]':
                self.attrs[key] = [] 
                value_str_list = value_str.split(',')
                for value_str_item in value_str_list:
                    self.attrs[key].append(float(value_str_item))
            elif type_str == '[string]':
                self.attrs[key] = [] 
                value_str_list = value_str.split(',')
                for value_str_item in value_str_list:
                    self.attrs[key].append(value_str_item)


    def __parse__(self):
        self.__parseInput__()
        self.__parseOutput__()
        self.__parseAttributes__()
        self.__byPassLegacy__()
