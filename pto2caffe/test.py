import os
import sys

ext2platform = {'pb':'tensorflow', 'tflite': 'tflite', 'onnx': 'onnx'}
exts = ['pb', 'tflite', 'onnx']

error_log = open('error_log', 'w+')
def search_model(path, exts):
    assert(os.path.isdir(path))

    models = []
    current_folder = os.walk(path)
    for path, dir_list, file_list in current_folder:
        for file_name in file_list:
            if file_name.split('.')[-1].lower() in exts:
                file_abs = path+'/'+file_name
                models.append(file_abs)

    return models 


def run_convert(model_path):
    platform = ext2platform[model_path.split('.')[-1].lower()]
    root_folder = ' -root_folder=/workspace/trunk/vc0768/tools/python_mc/pto2caffe/assets/data/bin_2560x1440/ '
    mean_std_scale = '-mean 0.485 0.456 0.406 -scale 255 255 255 -std 0.229 0.224 0.225 '
    bin_shape = '-bin_shape 3 2560 1440 '
    appex = '-compare=1 -auto_crop=1 '
    simplifier = '-simplifier=1'
    command = 'python convert.py -platform=' + platform + ' -model=' + model_path + root_folder + mean_std_scale + bin_shape + appex + simplifier

    ret = os.system(command)
    if ret != 0:
        print('Error:', command, '\n')
    else:
        print('Success:', command, '\n')

    return ret


def main():
    if len(sys.argv) >= 2:
        search_path = sys.argv[1]
    else:
        sys.exit('Input model path.')

    if len(sys.argv) >= 3 and sys.argv[2] in exts:
        models = search_model(search_path, [sys.argv[2]])
    else:
        models = search_model(search_path, exts)

    success = 0
    error = 0
    for model in models:
        ret = run_convert(model)
        if ret == 0: 
            success = success + 1 
        else:
            error = error + 1

    print('Total:', len(models), 'Success:', success, 'Error:', error)


if __name__ == "__main__":
    sys.exit(main())
