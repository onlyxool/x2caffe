import os
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
