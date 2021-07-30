import os
import sys
envroot = os.environ.get('MCHOME', os.environ['PWD'])
sys.path.append(envroot + 'toolchain/caffe2npu')
from compare_data import *
from load_data import LoadData

def compare(platform, model_name, input_file_name):
    print('Comparing...')
    dump_path = envroot + '/dump/' + model_name
    target_data = dump_path +'/'+ platform +'/'+ input_file_name
    caffe_data =  dump_path + '/caffe/' + input_file_name
    data_load = LoadData()

    print(target_data)
    print(caffe_data)
    if data_load.load_file(target_data, caffe_data) == False:
        print("Input directories are not match, please check")

    while 1:
        file1, file2 = data_load.get_one_batch()
        if file1 == None or file2 == None:
            print('Dump File not found.')
            break

        array1 = np.asarray(read_file(file1))
        array2 = np.asarray(read_file(file2))
        if len(array1) != len(array2):
            print("Get data error while calculate cosine similarity, lenght of 2 arrays are not equire!")
            continue

        cosin_simi = calc_cosine_simi(array1, array2) # close 1 is better
        max_err = calc_max_diff(array1, array2) # close 0 is better
        max_ratio = calc_max_ratio(array1, array2) # close 0 is better 

        layer_name = os.path.split(file1)[1].split('_')[1]
        print(layer_name, ':')
        print('  cosin_simi: %8f'% cosin_simi)
        print('  cmax_err: %8f'% max_err)
        print('  max_ratio: %8f'% max_ratio, '\n')
