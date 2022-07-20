import numpy as np
from tensorflow2caffe.op.operator import Operator


class Random(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('RandomStandardNormal'))
        self.setInited()


    def parse(self):
        self.layer_type = self.operator_code
        super().__parse__()

        if self.operator_code == 'RandomStandardNormal':
            if self.attrs['seed'] == 0 and self.attrs['seed2'] == 0 and self.attrs['dtype'] == 1:
                self.model.constant[self.outputs[0]] = np.random.normal(loc=0, scale=1, size=list(self.inputs_buf[0]))
            else:
                import sys
                errorMsg = 'Error: Operator [ RandomStandardNormal ] does not Support (seed =' + self.attrs['seed'] + ' or seed2 = ' + self.attrs['seed2'] + ').\n'
                sys.exit(errorMsg)
        else:
            raise NotImplementedError


    def convert(self):
        pass
