
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def quantile_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single quantile, keepdims=False
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array(0.5, dtype=np.float32)
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, multiple quantiles, axis=0, keepdims=True
    a = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
    q = np.array([0.25, 0.75], dtype=np.float32)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1, keepdims=False, float64
    a = np.array([[-5.0, 0.0, 5.0], [10.0, -10.0, 2.0]], dtype=np.float64)
    q = np.array(0.2, dtype=np.float64)
    weights = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with negative values, extreme quantiles [0.0, 1.0]
    a = np.array([-10.0, -2.5, -0.5, 4.0, 12.0], dtype=np.float32)
    q = np.array([0.0, 1.0], dtype=np.float32)
    weights = np.array([1.5, 0.5, 2.0, 1.0, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": -1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=0, keepdims=True
    np.random.seed(42)
    a = np.random.randn(4, 3, 2).astype(np.float32)
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    weights = np.random.uniform(0.1, 2.0, size=a.shape).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=1, keepdims=False
    a = np.random.uniform(-100.0, 100.0, size=(2, 5, 3)).astype(np.float64)
    q = np.array(0.75, dtype=np.float64)
    weights = np.ones((2, 5, 3), dtype=np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, negative axis=-1, keepdims=True
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 12.0
    q = np.array([0.33, 0.67], dtype=np.float32)
    weights = np.ones((2, 3, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": -1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, axis=2, keepdims=False
    a = np.random.randn(2, 3, 4, 2).astype(np.float32)
    q = np.array(0.5, dtype=np.float32)
    weights = np.random.uniform(0.5, 1.5, size=a.shape).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 2,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, multiple quantiles, weights matching array shape
    a = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    weights = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 3.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, int-like float input, multiple quantiles
    a = np.array([-50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0], dtype=np.float64)
    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.quantile_1"] = quantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.quantile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.quantile_1'.")


check_valid('jax.numpy.quantile', generated_inputs['jax.numpy.quantile_1'], lib="jax", suffix=1)
