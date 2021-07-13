import tflite
from tflite2caffe.model import Model

def convert(tf_file, input_tensor, caffe_model_name, caffe_model_path, dump_level=-1, param=None):
#print(tf_file, input_tensor, caffe_model_name, caffe_model_path, dump_level)
    with open(tf_file, 'rb') as f:
        buf = f.read()
        tfmodel = tflite.Model.GetRootAsModel(buf, 0)
    assert(tfmodel.Version() == 3)

    model = Model(tfmodel, param)
    model.convert()
    model.save(caffe_model_name, caffe_model_path)

