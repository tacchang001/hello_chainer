
# 環境

  * https://github.com/fixstars/clpy

SDKを入れたら動かなくなった（のでシステムをリストアする羽目に）

  
## OpenCL

  * https://github.com/intel/compute-runtime

標準のドライバが使える気がするが、cl.hは見つかるが、libOpenCL.soは見つからない。

```bash
$ clinfo --list

beignet-opencl-icd: no supported GPU found, this is probably the wrong opencl-icd package for this hardware
(If you have multiple ICDs installed and OpenCL works, you can ignore this message)
beignet-opencl-icd: no supported GPU found, this is probably the wrong opencl-icd package for this hardware
(If you have multiple ICDs installed and OpenCL works, you can ignore this message)
beignet-opencl-icd: no supported GPU found, this is probably the wrong opencl-icd package for this hardware
(If you have multiple ICDs installed and OpenCL works, you can ignore this message)
Platform #0: Intel(R) OpenCL HD Graphics
 `-- Device #0: Intel(R) Gen9 HD Graphics NEO
Platform #1: Intel Gen OCL Driver

```

## LLVM/Clang

```bash
$ sudo apt install clang libclang-dev
```

## CLBlast

```bash
$ mkdir build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
$ make -j4
$ make install

```

## Clpy

```bash
$ pip install cython
$ python setup.py install
```