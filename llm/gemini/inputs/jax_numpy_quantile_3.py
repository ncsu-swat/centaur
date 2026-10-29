
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def quantile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, median (q=0.5), axis 0
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.5,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, q=0.0 (min), keepdims=True
    a = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.0,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64 array, q=1.0 (max), axis=-1
    a = np.linspace(-100.0, 100.0, 21, dtype=np.float64)
    weights = np.ones(21, dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 1.0,
        "axis": -1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, axis 0, q=0.25
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    weights = np.ones((3, 3), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.25,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, axis 1, q=0.75, keepdims=True
    a = np.array([[-3.0, 0.0, 3.0, 6.0], [1.0, 4.0, 7.0, 10.0]], dtype=np.float32)
    weights = np.array([[1.0, 2.0, 1.0, 1.0], [2.0, 1.0, 3.0, 1.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.75,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int64 array with negative axis, q=0.5
    a = np.arange(-12, 12, dtype=np.int64).reshape(4, 6)
    weights = np.ones((4, 6), dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 0.5,
        "axis": -1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, axis 0, q=0.1
    a = np.random.uniform(-5.0, 5.0, size=(4, 3, 2)).astype(np.float32)
    weights = np.random.uniform(0.5, 2.0, size=(4, 3, 2)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.1,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 array, axis 1, q=0.9, keepdims=True
    a = np.random.randn(2, 5, 3).astype(np.float64)
    weights = np.ones((2, 5, 3), dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 0.9,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, axis 2, q=0.33
    a = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    weights = np.full((2, 3, 4), 2.0, dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.33,
        "axis": 2,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, axis -2, q=0.6, keepdims=True
    a = np.random.randn(2, 3, 4, 2).astype(np.float32)
    weights = np.ones((2, 3, 4, 2), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.6,
        "axis": -2,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.quantile_3"] = quantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.quantile_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.quantile_3'.")


check_valid('jax.numpy.quantile', generated_inputs['jax.numpy.quantile_3'], lib="jax", suffix=3)
