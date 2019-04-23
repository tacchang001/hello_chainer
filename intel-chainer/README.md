
# 環境

  * https://qiita.com/koreyou/items/888e8d65849a4f44d4f7
  * https://qiita.com/f0o0o/items/69d9b766008091a6e698
  * https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo
  * https://docs.chainer.org/en/stable/tips.html#how-do-i-accelerate-my-model-using-chainer-backend-for-intel-architecture
  * https://github.com/intel/mkl-dnn
 
## Intel MKL

  * https://github.com/intel/mkl-dnn

## Intel Chainer

```bash
$ git clone https://github.com/intel/chainer.git
$ pip install --no-binary :all: numpy
$ pip install --no-binary :all: scipy
$ pip install ideep4py
$ cd chainer
$ python3 setup.py install
```

## Fortanコンパイラ


```bash
    copying scipy/_lib/_numpy_compat.py -> build/lib.linux-x86_64-3.6/scipy/_lib
    running build_clib
    customize UnixCCompiler
    customize UnixCCompiler using build_clib
    building 'dfftpack' library
    error: library dfftpack has Fortran sources but no Fortran compiler found
```

```bash
sudo apt-get install gfortran
```

### 出典

  * https://stackoverflow.com/questions/29586487/still-cant-install-scipy-due-to-missing-fortran-compiler-after-brew-install-gcc

