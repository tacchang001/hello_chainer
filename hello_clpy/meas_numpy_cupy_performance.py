import time

# python -m clpy meas_numpy_cupy_performance.py -g0
#  ↓
# NotImplementedError: clpy does not supoort this

import cupy
import numpy

cnt = 100
N = 10
shapes = ((N, N), (N, N, N, N), (N, N, N, N, N, N), (N, N, N, N, N, N, N))


def meas_cupy(func, operand):
    stream = cupy.backend.Stream.null
    start = stream.record()
    for i in range(cnt):
        func(operand, xp=cupy)
    end = stream.record()
    end.synchronize()
    elapsed = cupy.backend.get_elapsed_time(start, end) / cnt
    return elapsed


def meas_numpy(func, operand):
    start = time.time()
    for i in range(cnt):
        func(operand, xp=numpy)
    end = time.time()
    elapsed = (end - start) * 1000 / cnt
    return elapsed


def meas_numpy_cupy(func):
    meas_results = []
    print("Array size,Numpy,Cupy")
    for shape in shapes:
        A = cupy.random.rand(*shape)
        a = A.get()
        cupy_elapsed = meas_cupy(func, A)
        numpy_elapsed = meas_numpy(func, a)
        meas_results.append((a.size, numpy_elapsed, cupy_elapsed))
    for result in meas_results:
        print("{0},{1},{2}".format(*result))


def array_add(A, xp):
    return A + A


print("Array add")
meas_numpy_cupy(array_add)


def array_sub(A, xp):
    return A - A


print("Array sub")
meas_numpy_cupy(array_sub)


def array_sum(A, xp):
    return A.sum()


print("Array sum")
meas_numpy_cupy(array_sum)


def array_argmax(A, xp):
    return A.argmax()


print("Array argmax")
meas_numpy_cupy(array_argmax)


def array_sort(A, xp):
    return xp.sort(A)


print("Array sort")
meas_numpy_cupy(array_sort)

shapes = ((N, N), (N, N, N, N), (N, N, N, N, N, N))


def array_tensordot(A, xp):
    return xp.tensordot(A, A)


print("Array tensordot")
meas_numpy_cupy(array_tensordot)


def array_matmul(A, xp):
    return xp.matmul(A, A)


print("Array matmul")
meas_numpy_cupy(array_matmul)


def array_einsum(A, xp):
    return xp.einsum("...i, ...j->...ij", A, A)


print("Array einsum")
meas_numpy_cupy(array_einsum)


def array_moveaxis(A, xp):
    return xp.moveaxis(A, 0, -1)


print("Array transpose")
meas_numpy_cupy(array_moveaxis)


def array_sin(A, xp):
    return xp.sin(A)


print("Array sin")
meas_numpy_cupy(array_sin)

shapes = ((N, N), (N * N, N * N), (N * N * N, N * N * N))


def array_eigh(A, xp):
    return xp.linalg.eigh(A)


print("Array eigenvalue")
meas_numpy_cupy(array_eigh)


def array_inv(A, xp):
    return xp.linalg.inv(A)


print("Array inv")
meas_numpy_cupy(array_inv)
