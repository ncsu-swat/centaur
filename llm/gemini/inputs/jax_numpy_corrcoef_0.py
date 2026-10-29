
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def corrcoef_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, rowvar=True, float32
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": True,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, rowvar=False, float64
    x = np.array([-2.5, 0.0, 1.5, 3.0], dtype=np.float64)
    y = np.array([1.2, -0.5, 2.1, 4.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": False,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, rowvar=True, float32
    x = np.array(
        [[1.0, -1.0, 2.0, 0.5], [0.0, 2.0, -1.0, 1.0]], dtype=np.float32
    )
    y = np.array([[3.0, 1.0, -2.0, 0.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": True,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, rowvar=False, float64
    # (N=4 observations, M=2 variables for x, K=1 variable for y)
    x = np.array(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]], dtype=np.float64
    )
    y = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": False,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex inputs, rowvar=True, complex64
    x = np.array(
        [[1.0 + 2.0j, -1.0 + 0.5j, 2.0 - 1.0j], [0.5 - 0.5j, 1.0 + 1.0j, -2.0 + 0.0j]],
        dtype=np.complex64,
    )
    y = np.array(
        [[2.0 + 1.0j, 0.0 - 1.0j, -1.0 + 2.0j]],
        dtype=np.complex64,
    )
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": True,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex inputs, rowvar=False, complex128
    x = np.array(
        [[1.0 + 1.0j, 2.0 - 1.0j], [3.0 + 0.0j, -1.0 + 2.0j], [0.0 - 2.0j, 1.0 + 1.0j]],
        dtype=np.complex128,
    )
    y = np.array(
        [[2.0 + 0.5j, 1.0 - 0.5j], [0.0 + 1.5j, -2.0 - 1.0j], [1.0 - 1.0j, 0.5 + 2.0j]],
        dtype=np.complex128,
    )
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": False,
        "dtype": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger random 2D arrays, rowvar=True, float64
    np.random.seed(42)
    x = np.random.randn(5, 50).astype(np.float64)
    y = np.random.randn(3, 50).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": True,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger random 2D arrays, rowvar=False, float32
    np.random.seed(123)
    x = np.random.randn(100, 4).astype(np.float32)
    y = np.random.randn(100, 2).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": False,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-variable 2D arrays, rowvar=True, float32
    x = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=np.float32)
    y = np.array([[15.0, 25.0, 35.0, 45.0, 55.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": True,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative and positive integers cast to float, rowvar=True, float64
    x = np.array([[-10, 0, 10, 20, -5], [5, -15, 25, -30, 0]], dtype=np.float64)
    y = np.array([[1, -2, 3, -4, 5], [2, 4, 6, 8, 10]], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "rowvar": True,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.corrcoef"] = corrcoef_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.corrcoef' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.corrcoef'.")


check_valid('jax.numpy.corrcoef', generated_inputs['jax.numpy.corrcoef'], lib="jax", suffix=0)
