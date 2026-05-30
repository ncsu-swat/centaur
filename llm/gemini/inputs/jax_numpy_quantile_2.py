
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def quantile_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 3 quantiles, no keepdims
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float64, with negative values, keepdims=True
    a = np.random.uniform(-10.0, 10.0, size=(3, 4)).astype(np.float64)
    q = np.array([0.5], dtype=np.float64)
    weights = np.random.uniform(0.1, 1.0, size=(3, 4)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, integer values, axis tuple (0, 2), keepdims=False
    a = np.random.randint(-50, 50, size=(2, 3, 4)).astype(np.int32)
    q = np.array([0.1, 0.9], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (0, 2),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, 4 quantiles, keepdims=True
    a = np.arange(10).astype(np.float32)
    q = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    weights = np.ones((10,), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, broadcasting weights (1D weights broadcasted to 2D)
    a = np.random.randn(5, 5).astype(np.float32)
    q = np.array([0.5], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(5,)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, axis (2, 3), keepdims=True
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    q = np.array([0.3, 0.6, 0.9], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(2, 2, 3, 3)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (2, 3),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, float64, single quantile
    a = np.random.randn(8).astype(np.float64)
    q = np.array([0.75], dtype=np.float64)
    weights = np.random.uniform(0.5, 2.0, size=(8,)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int64 array, float64 quantiles and weights
    a = np.random.randint(0, 100, size=(4, 2)).astype(np.int64)
    q = np.array([0.2, 0.8], dtype=np.float64)
    weights = np.random.uniform(1.0, 5.0, size=(4, 2)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with dimension size 1, axis (0, 2)
    a = np.random.randn(3, 1, 4).astype(np.float32)
    q = np.array([0.5], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(3, 1, 4)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (0, 2),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, 5 quantiles, axis (1,), keepdims=True
    a = np.random.randn(6, 6).astype(np.float32)
    q = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(6, 6)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.quantile_2"] = quantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.quantile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.quantile_2'.")


check_valid('jax.numpy.quantile', generated_inputs['jax.numpy.quantile_2'], lib="jax", suffix=2)
