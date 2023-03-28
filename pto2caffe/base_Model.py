from base import Base
from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer

class BaseModel(Base):
    def __init__(self, model, graph, param):
        super().__init__(model, graph)
        self.param = param
        self.layout = param['layout']
        self.platform = param['platform']

        self.pad = dict()
        self.layers = list()
        self.constant = dict()
        self.errorMsg = list()
        self.indentity = dict()
        self.operators = list()
        self.unsupport = list()
        self.tensor_dtype = dict()
        self.tensor_shape = dict()


    def convert(self):
        print("Converting the Caffe Model...")

        for index, input_name in enumerate(self.inputs):
            self.layers.append(make_caffe_input_layer(input_name, self.inputs_shape[index], index, self.param))

        for op in self.operators:
            self.layers.extend(op.convert())

        self.setConverted()


    def save(self, caffe_model_path):
        return save_caffe_model(caffe_model_path, self.layers)
