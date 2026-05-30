
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_nanpercentile_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, axis=(0,)
    a = np.random.randn(5, 5).astype(np.float32)
    a[1, 2] = np.nan
    a[3, 0] = np.nan
    q = np.array([50.0], dtype=np.float32)
    weights = np.random.rand(5, 5).astype(np.float32) + 0.1
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

    # Input 2: 3D array, float32, axis=(1,)
    a = np.random.randn(3, 4, 5).astype(np.float32)
    a[0, 2, 3] = np.nan
    a[2, 1, 1] = np.nan
    q = np.array([25.0, 75.0], dtype=np.float32)
    weights = np.ones((3, 4, 5), dtype=np.float32)
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

    # Input 3: 1D array, float64, axis=(0,)
    a = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    q = np.array([10.0, 50.0, 90.0], dtype=np.float64)
    weights = np.random.rand(10).astype(np.float64) + 0.5
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

    # Input 4: 3D array, float32, axis=(0, 2)
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[1, 1, 1] = np.nan
    q = np.array([50.0], dtype=np.float32)
    weights = np.random.rand(2, 3, 4).astype(np.float32) + 0.1
    input_dict = {
        "a": a,
        "q": q,
        "axis": (0, 2),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, float32, axis=(1,)
    a = np.random.randn(4, 4).astype(np.float32)
    a[0, 0] = np.nan
    a[3, 3] = np.nan
    q = np.array([0.0, 100.0], dtype=np.float32)
    weights = np.ones((4, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, float64, axis=(0,)
    a = np.random.randn(6, 2).astype(np.float64)
    a[2, 0] = np.nan
    a[4, 1] = np.nan
    q = np.array([33.3], dtype=np.float64)
    weights = np.ones((6, 2), dtype=np.float64)
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

    # Input 7: 3D array, float32, axis=(1, 2)
    a = np.random.randn(2, 2, 2).astype(np.float32)
    a[0, 1, 0] = np.nan
    q = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float32)
    weights = np.random.rand(2, 2, 2).astype(np.float32) + 1.0
    input_dict = {
        "a": a,
        "q": q,
        "axis": (1, 2),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, float32, axis=(0,)
    a = np.array([1.5, np.nan, 2.5, np.nan, 3.5, 4.5, 5.5, np.nan], dtype=np.float32)
    q = np.array([0.0], dtype=np.float32)
    weights = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
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

    # Input 9: 2D array, float64, axis=(0, 1)
    a = np.random.randn(3, 3).astype(np.float64)
    a[1, 1] = np.nan
    q = np.array([100.0], dtype=np.float64)
    weights = np.ones((3, 3), dtype=np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": (0, 1),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, float32, axis=(2,)
    a = np.random.randn(5, 2, 3).astype(np.float32)
    a[2, 0, 1] = np.nan
    a[4, 1, 2] = np.nan
    q = np.array([15.0, 85.0], dtype=np.float32)
    weights = np.random.rand(5, 2, 3).astype(np.float32) + 0.5
    input_dict = {
        "a": a,
        "q": q,
        "axis": (2,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanpercentile_2"] = jax_numpy_nanpercentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanpercentile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanpercentile_2'.")


check_valid('jax.numpy.nanpercentile', generated_inputs['jax.numpy.nanpercentile_2'], lib="jax", suffix=2)
