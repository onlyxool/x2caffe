import onnx
from onnx import shape_inference
from onnx2caffe.model import Model
from caffe_dump import dump_caffe_model
from compare import compare

def convert(onnx_file, input_tensor, caffe_model_path, dump_level=-1, param=None):
    onnx_model = onnx.load(onnx_file)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)

    model = Model(onnx_model, param)
    model.parse()
    model.convert()
    model.save(caffe_model_path)

    if dump_level >= 0:
        model.dump(onnx_model, param['model_name'], input_tensor, dump_level)

    if dump_level == 3:
        dump_caffe_model(caffe_model_path, input_tensor, param['input_file'])

    compare('onnx', onnx_model, caffe_model_path, input_tensor, param.get('compare', -1))
