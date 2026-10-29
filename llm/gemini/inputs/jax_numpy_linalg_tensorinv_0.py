
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tensorinv_inputs():
    list_of_inputs = []

    def make_invertible_tensor(shape, ind, dtype=np.float32):
        n = int(np.prod(shape[:ind]))
        if np.issubdtype(dtype, np.complexfloating):
            mat = (
                np.random.randn(n, n) + 1j * np.random.randn(n, n)
            ).astype(dtype)
        else:
            mat = np.random.randn(n, n).astype(dtype)
        mat += np.eye(n, dtype=dtype) * (n + 2.0)
        return mat.reshape(shape)

    # Input 1: 2D square matrix with ind=1, float32
    a = make_invertible_tensor((4, 4), 1, np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 1}))

    # Input 2: 3D tensor with ind=2, float32
    a = make_invertible_tensor((2, 2, 4), 2, np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 2}))

    # Input 3: 3D tensor with ind=2, float64
    a = make_invertible_tensor((2, 3, 6), 2, np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 2}))

    # Input 4: 4D tensor with ind=2, float32
    a = make_invertible_tensor((2, 2, 2, 2), 2, np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 2}))

    # Input 5: 4D tensor with ind=3, float64
    a = make_invertible_tensor((2, 2, 2, 8), 3, np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 3}))

    # Input 6: 4D tensor with ind=3, float32, including dimension of size 1
    a = make_invertible_tensor((2, 1, 3, 6), 3, np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 3}))

    # Input 7: 4D tensor with ind=2, float32, unequal sizes per half
    a = make_invertible_tensor((3, 4, 2, 6), 2, np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 2}))

    # Input 8: 2D square matrix with ind=1, complex64
    a = make_invertible_tensor((5, 5), 1, np.complex64)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 1}))

    # Input 9: 4D tensor with ind=2, complex128
    a = make_invertible_tensor((2, 3, 1, 6), 2, np.complex128)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 2}))

    # Input 10: 5D tensor with ind=3, float32
    a = make_invertible_tensor((2, 2, 3, 3, 4), 3, np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "ind": 3}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.tensorinv"] = tensorinv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.tensorinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.tensorinv'.")


check_valid('jax.numpy.linalg.tensorinv', generated_inputs['jax.numpy.linalg.tensorinv'], lib="jax", suffix=0)
