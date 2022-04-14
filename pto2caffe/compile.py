import os
import shutil
import py_compile

current_folder = os.walk(os.environ['PWD'])
current_folder_name = os.environ['PWD'].split('/')[-1]


for path, dir_list, file_list in current_folder:
    for file_name in file_list:
        ext = file_name.split('.')[-1]
        if ext == 'py':
            pyc_file = path.replace(current_folder_name, current_folder_name+'/release', 1) +'/'+ file_name.split('.')[0] + '.pyc'
            py_file = os.path.join(path, file_name)
            py_compile.compile(file=py_file, cfile=r'{}'.format(pyc_file), optimize=-1)



# PNNX
pnnx_dir = 'pytorch2caffe/pnnx/'
pnnx_build_dir = pnnx_dir + 'build/'
if os.path.isdir(pnnx_build_dir):
    shutil.rmtree(pnnx_build_dir)
os.mkdir(pnnx_build_dir)
os.system('cd pytorch2caffe/pnnx/build && cmake .. && make -j8 && cd ../../..')
shutil.copyfile('pytorch2caffe/pnnx/build/src/libpnnx.so', 'pytorch2caffe/libpnnx.so')
shutil.copyfile('pytorch2caffe/pnnx/build/src/libpnnx.so', '../release/pto2caffe/pytorch2caffe/libpnnx.so')
os.system('cd ../release/pto2caffe/ && ln -s ../caffe/python/caffe/ caffe')

print('Compile Pto2Caffe Done')
