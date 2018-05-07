from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) * B)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(0, A.dtype)
    C = tvm.compute(A.shape, lambda *i: tvm.max(B, A(*i)))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    x = tvm.placeholder(shape, dtype=dtype, name="x")
    grad_x = tvm.placeholder(shape, dtype=dtype, name="grad_x")
    zero = tvm.const(0, x.dtype)

    y = tvm.compute(x.shape, lambda *i: tvm.select(x(*i) > 0, grad_x(*i), 0.0))
    s = tvm.create_schedule(y.op)
    f = tvm.build(s, [x, grad_x, y], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")

    if not transposeA and not transposeB:
        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.compute((shapeA[0], shapeB[1]), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    elif transposeA and transposeB:
        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.compute((shapeA[1], shapeB[0]), lambda i, j: tvm.sum(A[k, i] * B[j, k], axis=k), name='C')
    elif transposeA:
        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.compute((shapeA[1], shapeB[1]), lambda i, j: tvm.sum(A[k, i] * B[k, j], axis=k), name='C')
    else:
        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.compute((shapeA[0], shapeB[0]), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    v1 = 8
    v2 = 8
    v3 = 4
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], v1, v2)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=v3)
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)

    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    shapeY = (N, M, H - R + 1, W - S + 1)
    X = tvm.placeholder(shapeX, dtype=dtype, name='X')
    F = tvm.placeholder(shapeF, dtype=dtype, name='F')

    c = tvm.reduce_axis((0, C), name='c')
    rh = tvm.reduce_axis((0, R), name='r')
    sw = tvm.reduce_axis((0, S), name='s')
    
    Y = tvm.compute(shapeY, lambda n, m, h, w: tvm.sum(X[n, c, h + rh, w + sw] * F[m, c, rh, sw], axis=[c, rh, sw]), name='Y')
    s = tvm.create_schedule(Y.op)
    
    f = tvm.build(s, [X, F, Y], tgt, target_host=tgt_host, name=func_name)
    return f

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    X = tvm.placeholder(shape, dtype=dtype, name='X')
    x_j = tvm.reduce_axis((0, shape[0]), name='x_j')
    max_x = tvm.compute((shape[0],), lambda i: tvm.max(X[i, x_j], axis=x_j), name = 'max_x')
    e_x = tvm.compute(shape, lambda i, j: tvm.exp(X[i, j] - max_x[i]), name='e_x')
    sum_x = tvm.compute((shape[0],), lambda i: tvm.sum(X[i, x_j], axis=x_j), name = 'sum_x')
    Y = tvm.compute(shape, lambda i, j: e_x[i, j] / sum_x[i], name='Y')

    s = tvm.create_schedule(Y.op)
    
    f = tvm.build(s, [X, Y], tgt, target_host=tgt_host, name=func_name)
    return f

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f