#!/usr/bin/env python3

import os
import sys
import glob
import shutil
import argparse
import py_compile
import multiprocessing as mp


# date:   2022.08.26
# author: duruyao@gmail.com
# desc:   build model converter
# usage:  python3 build.py --help

def info_ln(values):
    print('\033[1;32;32m', values, '\033[m', sep='', end='\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_type', required=False,
                        default='release', type=str, choices=['release', 'debug'],
                        help='set build type (default: release)')
    parser.add_argument('-j', '--jobs', required=False,
                        default=int(mp.cpu_count() / 2), type=int,
                        help=f'set the number of threads to use (default: {int(mp.cpu_count() / 2)})')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--is_vc0728', action="store_true",
                       help=f'build for vc0728')
    group.add_argument('--is_vc0768', action="store_true",
                       help=f'build for vc0768')
    group.add_argument('--is_vc0778', action="store_true",
                       help=f'build for vc0778')

    args = parser.parse_args()

    build_type = str(args.build_type)  # {debug|release}
    cmake_build_type = build_type.capitalize()  # {Debug|Release}
    jobs = min(mp.cpu_count() - 1, args.jobs)  # number of working threads
    is_vc0728 = args.is_vc0728  # {False|True}
    is_vc0768 = args.is_vc0768  # {False|True}
    is_vc0778 = args.is_vc0778  # {False|True}

    current_dir = os.getcwd()
    project_root = os.path.dirname(os.path.realpath(sys.argv[0]))
    target_dir = f'{current_dir}/build_{build_type}'
    shutil.rmtree(target_dir, ignore_errors=True)

    # compile py to pyc
    py_base_dir = f'{project_root}/pto2caffe'
    for py_src_path in glob.glob(f'{py_base_dir}/**/*.py', recursive=True):
        py_dst_path = f'{target_dir}/{os.path.relpath(py_src_path, py_base_dir)}'
        pyc_dst_path = f'{target_dir}/{os.path.relpath(py_src_path, py_base_dir)}c'
        if 'release' == build_type:
            py_compile.compile(py_src_path, pyc_dst_path, optimize=-1)
        elif 'debug' == build_type:
            os.makedirs(os.path.dirname(py_dst_path), exist_ok=True)
            shutil.copyfile(py_src_path, py_dst_path)

    # build pnnx
    pnnx_dir = f'{project_root}/pto2caffe/pytorch2caffe/pnnx'
    pnnx_build_dir = f'{pnnx_dir}/build-{build_type}'
    # shutil.rmtree(pnnx_build_dir, ignore_errors=True) # rm -rf build*/
    os.system(f'cmake -H{pnnx_dir} -B{pnnx_build_dir} -DCMAKE_BUILD_TYPE={cmake_build_type}')
    os.system(f'cmake --build {pnnx_build_dir} --target all -- -j {jobs}')
    shutil.copyfile(f'{pnnx_build_dir}/src/libpnnx.so', f'{target_dir}/pytorch2caffe/libpnnx.so')
    shutil.copyfile(f'{pnnx_build_dir}/src/libpnnx.so', f'{project_root}/pto2caffe/pytorch2caffe/libpnnx.so')

    info_ln(f'Installed the built Pto2Caffe to \'{target_dir}\'')


if __name__ == '__main__':
    main()
