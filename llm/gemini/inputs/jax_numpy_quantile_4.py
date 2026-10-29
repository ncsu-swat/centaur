
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def quantile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, median
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.5,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=(0,), keepdims=True
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    weights = np.ones((3, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.25,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=(1,), 75th percentile
    a = np.random.randn(4, 5).astype(np.float64)
    weights = np.random.uniform(0.1, 1.0, size=(4, 5)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": 0.75,
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values, axis=(-1,), q=0.0 (min)
    a = np.array([[-10.0, -5.0, -1.0], [-20.0, -15.0, -3.0]], dtype=np.float32)
    weights = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.0,
        "axis": (-1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: q=1.0 (max), 3D array
    a = np.random.randn(2, 3, 4).astype(np.float32)
    weights = np.random.uniform(0.5, 2.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 1.0,
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple axes in tuple
    a = np.random.uniform(-5.0, 5.0, size=(3, 4, 5)).astype(np.float32)
    weights = np.ones((3, 4, 5), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.35,
        "axis": (0, 2),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer array input
    a = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
    weights = np.array([1, 1, 2, 2, 1, 1], dtype=np.int32)
    input_dict = {
        "a": a,
        "q": 0.6,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, axis=(2,)
    a = np.random.randn(2, 2, 3, 4).astype(np.float32)
    weights = np.random.uniform(0.1, 5.0, size=(2, 2, 3, 4)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.8,
        "axis": (2,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, reduction over all dimensions via tuple
    a = np.arange(9, dtype=np.float64).reshape(3, 3)
    weights = np.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64
    )
    input_dict = {
        "a": a,
        "q": 0.5,
        "axis": (0, 1),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, same shape weights
    a = np.random.uniform(1.0, 10.0, size=(4, 6)).astype(np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(4, 6)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.1,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.quantile_4"] = quantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.quantile_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.quantile_4'.")


check_valid('jax.numpy.quantile', generated_inputs['jax.numpy.quantile_4'], lib="jax", suffix=4)
