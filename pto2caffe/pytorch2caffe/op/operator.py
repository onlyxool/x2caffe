from base import Base

class Operator(Base):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, None, index)
        self.operator = type_code
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
        return self.layer_type if self.layer_type is not None else self.operator


    @property
    def name(self):
        return self.pnnx.get_ops_name(self.index)


    @property
    def shorty(self):
        return '[%s](%s)' % (self.name, self.type)


    def str(self):
        return '[' + self.name + '] (' + self.type + ')'


    def __str__(self):
        inames = str([t for t in self.inputs])
        onames = str([t for t in self.outputs])
        return '%s attr%s: %s -> %s' % (self.shorty, self.attrs, inames, onames)


    def parseInput(self):
        self.inputs = self.pnnx.get_ops_inputs(self.index)
        for i, input_name in enumerate(self.inputs):
            self.inputs_shape.append(self.pnnx.get_ops_input_shape(self.index, i, input_name))
            tensor = self.pnnx.get_ops_attr(self.index, input_name)
            self.inputs_buf.append(tensor.reshape(self.inputs_shape[i]) if tensor is not None else None)


    def parseOutput(self):
        self.outputs = self.pnnx.get_ops_outputs(self.index)
        for i, output_name in enumerate(self.outputs):
            self.outputs_shape.append(self.pnnx.get_ops_output_shape(self.index, i))


    def parseAttributes(self):
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
