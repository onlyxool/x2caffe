import os
import sys

ext2platform = {'pb':'tensorflow', 'tflite': 'tflite', 'onnx': 'onnx', 'pt': 'pytorch'}
exts = ['pb', 'tflite', 'onnx', 'pt']


def search_model(path, exts):
    assert(os.path.isdir(path))

    models_path = list()
    models = list()
    current_folder = os.walk(path)
    for path, dir_list, file_list in current_folder:
        for file_name in file_list:
            if file_name.split('.')[-1].lower() in exts:
                file_abs = path+'/'+file_name
                models.append(file_name)
                models_path.append(file_abs)

    return models, models_path


def run_convert(model_path, ext, op):
    if op is not None and ext == 'onnx':
        import onnx
        onnx_model = onnx.load(model_path)
        for index, node in enumerate(onnx_model.graph.node):
            if node.op_type == op:
                break
        else:
            return 0

    platform = ext2platform[model_path.split('.')[-1].lower()]

    if platform == 'pytorch':
        appex = ' -compare=1 -crop_h=600 -crop_w=600 '
    else:
        appex = ' -compare=1 -auto_crop=1 '

    if platform == 'onnx':
        simplifier = '-simplifier=1'
    else:
        simplifier = ''

    command = 'python convert.py -platform=' + platform + ' -model=' + model_path + appex + simplifier

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
        models, models_path = search_model(search_path, [sys.argv[2]])
    else:
        models, models_path = search_model(search_path, exts)


    success = list()
    error = list()
    for index, model_path in enumerate(models_path):
        ret = run_convert(model_path, sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else None)

        success.append(models[index]) if ret == 0 else error.append(models[index])

    print(success)
    print(error)
    print('Total:', len(models_path), 'Success:', len(success), 'Error:', len(error))


if __name__ == "__main__":
    sys.exit(main())
