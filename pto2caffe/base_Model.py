from base import Base
from caffe_transform import save_caffe_model

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
        self.tensor_shape = dict()


    def save(self, caffe_model_path):
        return save_caffe_model(caffe_model_path, self.layers)
