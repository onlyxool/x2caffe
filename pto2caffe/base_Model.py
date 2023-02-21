from base import Base

class BaseModel(Base):
    def __init__(self, model, graph, param):
        super().__init__(model, graph)
        self.param = param
        self.layout = param['layout']

        self.pad = dict()
        self.shape = dict()
        self.layers = list()
        self.constant = dict()
        self.errorMsg = list()
        self.indentity = dict()
        self.operators = list()
        self.unsupport = list()
