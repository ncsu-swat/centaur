
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanquantile_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs, axis=(0,), median (q=0.5)
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 1.0, 3.0], dtype=np.float32)
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

    # Input 2: 1D array without NaNs, minimum (q=0.0), keepdims=True
    a = np.array([10.5, -3.2, 0.0, 4.8, -15.1], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 0.0,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with NaNs, axis=(0,), lower quartile (q=0.25)
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    weights = np.ones((2, 3), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.25,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, axis=(1,), upper quartile (q=0.75), keepdims=True
    a = np.array([[-5.0, -1.0, 0.0], [np.nan, 2.0, 8.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    weights = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0], [2.0, 2.0, 1.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.75,
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, axis=(-1,), maximum (q=1.0)
    a = np.array([[np.nan, 2.5], [1.2, -4.5]], dtype=np.float64)
    weights = np.array([[0.5, 1.5], [2.0, 1.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 1.0,
        "axis": (-1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array with NaNs, axis=(1,), q=0.3
    np.random.seed(42)
    a = np.random.randn(2, 4, 3).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    weights = np.random.uniform(0.5, 2.0, size=(2, 4, 3)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.3,
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, axis=(2,), keepdims=True, q=0.9
    a = np.random.uniform(-10.0, 10.0, size=(3, 2, 5)).astype(np.float64)
    a[a < -5.0] = np.nan
    weights = np.random.uniform(0.1, 1.0, size=(3, 2, 5)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": 0.9,
        "axis": (2,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, negative axis=(-2,), q=0.67
    a = np.array([[np.nan, 10.0], [5.0, np.nan], [2.0, 8.0]], dtype=np.float32)
    weights = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.67,
        "axis": (-2,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, axis=(3,), q=0.42
    a = np.random.randn(2, 2, 3, 4).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    weights = np.ones((2, 2, 3, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.42,
        "axis": (3,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with mixed values and weights, q=0.1
    a = np.array([-100.0, -50.0, 0.0, np.nan, 50.0, 100.0], dtype=np.float64)
    weights = np.array([5.0, 1.0, 2.0, 1.0, 1.0, 10.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 0.1,
        "axis": (-1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanquantile_4"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_4'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_4'], lib="jax", suffix=4)
