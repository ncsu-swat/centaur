
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def matrix_norm_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix, ord=1, keepdims=False
    x = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": False, "ord": int(1)})
    )

    # Input 2: 2D square float32 matrix, ord=2, keepdims=True
    x = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": True, "ord": int(2)})
    )

    # Input 3: 2D rectangular float64 matrix (M > N), ord=-1, keepdims=False
    x = np.array(
        [[1.5, -2.5], [3.0, 4.0], [-0.5, 2.0]], dtype=np.float64
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": False, "ord": int(-1)})
    )

    # Input 4: 2D rectangular float64 matrix (M < N), ord=-2, keepdims=True
    x = np.array(
        [[2.0, 1.0, -1.0], [0.0, 3.0, 2.0]], dtype=np.float64
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": True, "ord": int(-2)})
    )

    # Input 5: 3D batched float32 matrices, ord=1, keepdims=True
    x = np.random.randn(3, 4, 4).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": True, "ord": int(1)})
    )

    # Input 6: 3D batched float32 matrices, ord=2, keepdims=False
    x = np.random.randn(2, 3, 5).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": False, "ord": int(2)})
    )

    # Input 7: 3D batched complex64 matrices, ord=-1, keepdims=True
    x = (np.random.randn(4, 2, 2) + 1j * np.random.randn(4, 2, 2)).astype(
        np.complex64
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": True, "ord": int(-1)})
    )

    # Input 8: 3D batched complex128 matrices, ord=-2, keepdims=False
    x = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": False, "ord": int(-2)})
    )

    # Input 9: 4D batched float32 matrices, ord=1, keepdims=False
    x = np.random.randn(2, 2, 3, 4).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": False, "ord": int(1)})
    )

    # Input 10: 4D batched float64 matrices, ord=2, keepdims=True
    x = np.random.randn(1, 3, 4, 2).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": True, "ord": int(2)})
    )

    # Input 11: 2D integer-like values in float32 with negative elements, ord=-1, keepdims=False
    x = np.array([[-10.0, 5.0], [2.0, -8.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": False, "ord": int(-1)})
    )

    # Input 12: 5D batched float32 matrices, ord=-2, keepdims=True
    x = np.random.randn(2, 1, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "keepdims": True, "ord": int(-2)})
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.matrix_norm_2"] = matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matrix_norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matrix_norm_2'.")


check_valid('jax.numpy.linalg.matrix_norm', generated_inputs['jax.numpy.linalg.matrix_norm_2'], lib="jax", suffix=2)
