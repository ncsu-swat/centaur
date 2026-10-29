
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tensorsolve_inputs():
    list_of_inputs = []

    def make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float32):
        n = int(np.prod(b_shape))
        ndim = len(b_shape) + len(q_shape)

        allaxes = list(range(ndim))
        for k in axes:
            allaxes.remove(k)
            allaxes.append(k)

        inv_axes = np.argsort(allaxes)

        if np.issubdtype(dtype, np.complexfloating):
            m = np.eye(n, dtype=dtype) + 0.1 * (
                np.random.randn(n, n) + 1j * np.random.randn(n, n)
            ).astype(dtype)
        else:
            m = np.eye(n, dtype=dtype) + 0.1 * np.random.randn(n, n).astype(
                dtype
            )

        a_transposed = m.reshape(b_shape + q_shape)
        a = a_transposed.transpose(inv_axes)
        return a

    # Input 1: 1D b, 1D x, float32, axes=()
    b_shape = (4,)
    q_shape = (4,)
    axes = ()
    b = np.random.randn(*b_shape).astype(np.float32)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 2: 2D b, 1D x, float32, axes=()
    b_shape = (2, 2)
    q_shape = (4,)
    axes = ()
    b = np.random.randn(*b_shape).astype(np.float32)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 3: 2D b, 2D x, float64, axes=(0,)
    b_shape = (2, 3)
    q_shape = (3, 2)
    axes = (0,)
    b = np.random.randn(*b_shape).astype(np.float64)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 4: 1D b, 2D x, float32, axes=(1,)
    b_shape = (6,)
    q_shape = (2, 3)
    axes = (1,)
    b = np.random.randn(*b_shape).astype(np.float32)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 5: 3D b, 1D x, float32, axes=()
    b_shape = (2, 2, 2)
    q_shape = (8,)
    axes = ()
    b = np.random.randn(*b_shape).astype(np.float32)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 6: complex64, 2D b, 1D x, axes=()
    b_shape = (2, 2)
    q_shape = (4,)
    axes = ()
    b = (
        np.random.randn(*b_shape) + 1j * np.random.randn(*b_shape)
    ).astype(np.complex64)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 7: complex128, 1D b, 1D x, axes=(0,)
    b_shape = (5,)
    q_shape = (5,)
    axes = (0,)
    b = (
        np.random.randn(*b_shape) + 1j * np.random.randn(*b_shape)
    ).astype(np.complex128)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 8: 2D b, 2D x, multiple axes reordered, float32
    b_shape = (2, 2)
    q_shape = (2, 2)
    axes = (0, 2)
    b = np.random.randn(*b_shape).astype(np.float32)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 9: float64, 3D b, 2D x, axes=(1, 3)
    b_shape = (2, 1, 3)
    q_shape = (3, 2)
    axes = (1, 3)
    b = np.random.randn(*b_shape).astype(np.float64)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 10: 1D b, 3D x, float32, axes=(2,)
    b_shape = (8,)
    q_shape = (2, 2, 2)
    axes = (2,)
    b = -np.random.rand(*b_shape).astype(np.float32)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    # Input 11: 4D b, 1D x, float32, axes=()
    b_shape = (2, 1, 2, 2)
    q_shape = (8,)
    axes = ()
    b = np.random.randn(*b_shape).astype(np.float32)
    a = make_well_conditioned_a(b_shape, q_shape, axes, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "axes": axes}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.tensorsolve"] = tensorsolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.tensorsolve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.tensorsolve'.")


check_valid('jax.numpy.linalg.tensorsolve', generated_inputs['jax.numpy.linalg.tensorsolve'], lib="jax", suffix=0)
