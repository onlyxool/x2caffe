import tflite
from tflite2caffe.model import Model
from caffe_dump import dump_caffe_model
from compare import compare

def convert(tf_file, input_tensor, caffe_model_name, caffe_model_path, dump_level=-1, param=None):
    with open(tf_file, 'rb') as f:
        model_byte = f.read()
        tfmodel = tflite.Model.GetRootAsModel(model_byte, 0)
    assert(tfmodel.Version() == 3)

    model = Model(tfmodel, param)
    model.parse()

    model.convert()
    model.save(caffe_model_name, caffe_model_path)

    if dump_level >= 0:
        model.dump(model_byte, caffe_model_name, input_tensor, dump_level)

    if dump_level == 3:
        dump_caffe_model(caffe_model_name, caffe_model_path, input_tensor, param['input_file'])

    if param.get('compare', -1) == 1:
        compare('tflite', caffe_model_name, param['input_file'])
