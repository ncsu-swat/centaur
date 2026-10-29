
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def percentile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, median (q=50.0)
    a = np.array([1.0, 2.0, 5.0, 10.0, 20.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 50.0,
        "axis": (0,),
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array, 25th percentile along axis 1 with keepdims
    a = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)
    weights = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 25.0,
        "axis": (1,),
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array with negative values, 75th percentile along axis 0
    a = np.array([[-10, 5, 0], [-20, 15, -5]], dtype=np.int32)
    weights = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 75.0,
        "axis": (0,),
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, 0th percentile (min) along axis -1 with keepdims
    a = np.random.randn(2, 3, 4).astype(np.float32)
    weights = np.random.uniform(0.1, 2.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.0,
        "axis": (-1,),
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, 100th percentile (max) along axes (0, 1)
    a = np.random.uniform(-10.0, 10.0, size=(3, 4, 5)).astype(np.float32)
    weights = np.ones((3, 4, 5), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 100.0,
        "axis": (0, 1),
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array, 99.9th percentile
    a = np.arange(100, dtype=np.int64)
    weights = np.linspace(0.5, 5.0, 100, dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 99.9,
        "axis": (0,),
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, 10th percentile along multiple axes (1, 2) with keepdims
    a = np.random.randn(2, 3, 3, 2).astype(np.float32)
    weights = np.random.exponential(scale=1.0, size=(2, 3, 3, 2)).astype(
        np.float32
    ) + 0.01
    input_dict = {
        "a": a,
        "q": 10.0,
        "axis": (1, 2),
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array, 33.33th percentile along axis 0 with keepdims
    a = np.array([[-5.0, -1.0], [0.0, 3.0], [7.0, 12.0]], dtype=np.float32)
    weights = np.array([[2.0, 1.0], [1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 33.33,
        "axis": (0,),
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 array, 50th percentile along axis 1 without keepdims
    a = np.random.randn(4, 5, 2).astype(np.float64)
    weights = np.random.uniform(0.5, 3.0, size=(4, 5, 2)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": 50.0,
        "axis": (1,),
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array, 0.5th percentile with keepdims
    a = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.5,
        "axis": (0,),
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D float32 array, 80th percentile across all axes (0, 1)
    a = np.linspace(-3.0, 3.0, 12, dtype=np.float32).reshape(3, 4)
    weights = np.ones((3, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 80.0,
        "axis": (0, 1),
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.percentile_4"] = percentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.percentile_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.percentile_4'.")


check_valid('jax.numpy.percentile', generated_inputs['jax.numpy.percentile_4'], lib="jax", suffix=4)
