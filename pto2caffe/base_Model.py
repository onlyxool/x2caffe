from base import Base

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
