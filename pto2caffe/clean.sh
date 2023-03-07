# Remove all pyc and __pycache__
find . -name '*.pyc' -type f -print -exec rm {} \;
find . -name '__pycache__' -exec rmdir {} \;
rm .dbg_log.txt
rm pytorch2caffe/libpnnx.so
