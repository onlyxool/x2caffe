import os
import sys
import argparse

from preprocess import get_input_tensor
from preprocess import preprocess



def set_bin_shape(source_tensor, param):
    param['source_shape'] = [1] + list(source_tensor.shape)


def isContainFile(path, ext_list):
    if not os.path.isdir(path):
        return False

    datas = os.walk(path)
    for path_list, dir_list, file_list in datas:
        for file_name in file_list:
            if file_name.split('.')[-1].lower() in ext_list:
                return True

    return False


def RGB2BGR(param):
    if isinstance(param, list) and len(param) == 3:
        ret_param = []
        ret_param.append(param[2])
        ret_param.append(param[1])
        ret_param.append(param[0])
        return ret_param
    else:
        return param


def make_source_list(data_path, errorMsg):
    exts = ['jpg', 'jpeg', 'png', 'bmp', 'bin']

    source = os.path.abspath(data_path) + '/list.txt'
    source_file = open(source, 'w')

    found = False
    for file_name in os.listdir(data_path):
        if file_name.split('.')[-1].lower() in exts:
            source_file.writelines('%s\n'%file_name)
            found = True

    source_file.close()

    if not found:
        sys.exit(errorMsg + 'Can\'t find any data file or image file in ' + data_path)

    return source


def CheckParam(param):
    # platform
    errorMsg = '\nArgument Check Failed: '
    param['platform'] = param['platform'].lower()
    if param['platform'] not in ['tensorflow', 'pytorch', 'tflite', 'onnx']:
        errorMsg = errorMsg + 'argument -platform: invalid choice: ' + param['platform'] + ' (choose from TensorFlow, Pytorch, TFLite, ONNX)'
        sys.exit(errorMsg)

    # model
    if param['platform'] == 'tensorflow':
        if not os.path.isfile(param['model']) and not os.path.isdir(param['model']):
            errorMsg = errorMsg + 'Model File not exist  ' + param['model']
            sys.exit(errorMsg)
    elif not os.path.isfile(param['model']):
        errorMsg = errorMsg + 'Model File not exist  ' + param['model']
        sys.exit(errorMsg)
    param['model'] = os.path.abspath(os.path.normpath(param['model']))

    # root_folder
    if not os.path.isdir(param['root_folder']):
        errorMsg = errorMsg + 'Illegal root_folder  ' + param['root_folder']
        sys.exit(errorMsg)
    param['root_folder'] = os.path.abspath(os.path.normpath(param['root_folder']))

    # source
    if param.get('source', None) is not None:
        if os.path.isfile(param['source']):
            param['source'] = os.path.abspath(os.path.normpath(param['source']))
        else:
            param['source'] = make_source_list(param['root_folder'], errorMsg)
    else:
        param['source'] = make_source_list(param['root_folder'], errorMsg)

    # bin_shape & dtype
    if isContainFile(param['root_folder'], ['bin']):
        param['file_type'] = 'raw'
        if param['dtype'] is None:
            errorMsg = errorMsg + 'argument dtype can\'t be None'
            sys.exit(errorMsg)
        if param['bin_shape'] is None:
            errorMsg = errorMsg + 'argument bin_shape can\'t be None'
            sys.exit(errorMsg)
        elif len(param['bin_shape']) != 3:
            errorMsg = errorMsg + 'Bin file shape should be 3 dimension'
            sys.exit(errorMsg)
    elif isContainFile(param['root_folder'], ['jpg', 'bmp', 'png', 'jpeg']):
        param['file_type'] = 'img'
    else:
        errorMsg = errorMsg + 'Can\'t find any data file or image file in ' + param['root_folder']
        sys.exit(errorMsg)

    # BGR -> RGB
    if param['color_format'] == 'BGR':
        param['mean'] = RGB2BGR(param['mean'])
        param['std'] = RGB2BGR(param['std'])
        param['scale'] = RGB2BGR(param['scale'])

    if 'auto_crop' in param and param['auto_crop'] == 1:
        if param['platform'] == 'pytorch':
            errorMsg = errorMsg + 'Pytorch model support dynamic shape, auto_crop can\'t apply.'
            sys.exit(errorMsg)

    # Layout
    if 'layout' not in param or param['layout'] is None:
        if param['platform'] == 'tensorflow' or param['platform'] == 'tflite':
            param['layout'] = 'NHWC'
        else:
            param['layout'] = 'NCHW'


def Convert(param=None):
    # Set Log level
    os.environ['GLOG_minloglevel'] = str(param.get('log', 2)) # 0:DEBUG 1:INFO 2:WARNING 3:ERROR
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(param.get('log', 2)) # 0:INFO 1:WARNING 2:ERROR 3:FATAL

    # Check Param
    CheckParam(param)

    framework = param['platform']
    model_path = param['model']
    print(model_path)

    param['model_name'] = os.path.basename(model_path)
    caffe_model_path = os.path.splitext(model_path)[0]
    dump_level = param.get('dump', -1)

    input_tensor = get_input_tensor(param['root_folder'], param)

    if param['file_type'] == 'raw':
        set_bin_shape(input_tensor, param)

    if framework == 'pytorch':
        from pytorch2caffe.convert import convert as PytorchConvert
        PytorchConvert(model_path, input_tensor, caffe_model_path, dump_level, param)
    elif framework == 'tensorflow':
        from tensorflow2caffe.convert import convert as TensorflowConvert
        TensorflowConvert(model_path, input_tensor, caffe_model_path, dump_level, param)
    elif framework == 'tflite':
        from tflite2caffe.convert import convert as TensorLiteConvert
        TensorLiteConvert(model_path, input_tensor, caffe_model_path, dump_level, param)
    elif framework == 'onnx':
        from onnx2caffe.convert import convert as OnnxConvert
        OnnxConvert(model_path, input_tensor, caffe_model_path, dump_level, param)
    else:
        raise NotImplementedError


def args_():
    args = argparse.ArgumentParser(description='Tensorflow/Tensorflow lite/ONNX/Pytorch to Caffe Conversion Usage', epilog='')
    args.add_argument('-platform',      type = str,     required = True,
            help = 'Pytorch/Tensorflow/ONNX/TFLite')
    args.add_argument('-model',         type = str,     required = True,
            help = 'Orginal Model File')
    args.add_argument('-root_folder',   type = str,     required = True,
            help = 'Specify the Data root folder')
    args.add_argument('-source',        type = str,     required = False,
            help = 'Specify the data source')
    args.add_argument('-dtype',         type = str,     required = False,   choices=['u8', 's16', 'f32'],   default='u8',
            help = 'Specify the Data type, 0:u8 1:s16 2:f32')
    args.add_argument('-bin_shape',     type = int,     required = False,   nargs='+',
            help = 'Specify the Data shape when input data is bin file, default layout is [C,H,W]')
    args.add_argument('-color_format',  type = str,     required = False,   choices=['BGR', 'RGB', 'GRAY'],
            help = 'Specify the images color format, 0:BGR 1:RGB 2:GRAY')

#    args.add_argument('-input_shape',   type = int,     required = False,   nargs='+',
#            help = 'Model input shape')
    args.add_argument('-layout',        type = str,     required = False,   choices=['NCHW', 'NHWC'],
            help = 'Model input layout [NCHW NHWC]')

    args.add_argument('-scale',         type = float,   required = False,   nargs='+',      default = [1, 1, 1],
            help = 'scale value')
    args.add_argument('-mean',          type = float,   required = False,   nargs='+',
            help = 'mean value')
    args.add_argument('-std',           type = float,   required = False,   nargs='+',
            help = 'std value')
    args.add_argument('-crop_h',        type = int,     required = False,
            help = 'Specify if we would like to centrally crop input image')
    args.add_argument('-crop_w',        type = int,     required = False,
            help = 'Specify if we would like to centrally crop input image')
    args.add_argument('-auto_crop',     type = int,     required = False,   default=0,      choices=[0, 1],
            help = 'Crop the input data according to the model inputs size')
    args.add_argument('-dump',          type = int,     required = False,   default=-1,     choices=[0, 1, 2, 3],
            help = 'dump blob  1:print output.  2:print input & ouput')
    args.add_argument('-compare',       type = int,     required = False,   default=-1,     choices=[0, 1, 2],
            help = 'Compare network output, 0:Compare latest layer 1:Compare every layer')
    args.add_argument('-log',           type = int,     required = False,   default=2,      choices=[0, 1, 2],
            help = 'log print level, 0:Debug 1:Info 2:Warning, 3:ERROR')
    args.add_argument('-simplifier',    type = int,     required = False,   default=0,      choices=[0, 1],
            help = 'simplify onnx model by onnx-simplifier')
    args.add_argument('-streamlit',     type = int,     required = False,   default=0,      choices=[0, 1],
            help = 'Web Interface Flag')
    args = args.parse_args()
    return args


def main():
    print('\nModel Convertor:', end=' ')
    args = args_()
    Convert(args.__dict__)

    # Delete all argmuments
    for i in range(1, len(sys.argv)):
        sys.argv.pop()

    print('Conversion Done.\n')


if __name__ == "__main__":
    sys.exit(main())
