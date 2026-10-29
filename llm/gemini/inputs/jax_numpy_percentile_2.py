
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def percentile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, median (q=50.0), keepdims=False
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 1.0, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 50.0,
        "axis": 0,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis=1, q=25.0, keepdims=True
    a = np.random.uniform(0.0, 10.0, size=(4, 6)).astype(np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(4, 6)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 25.0,
        "axis": 1,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array, axis=0, q=75.0, keepdims=False
    a = np.random.randint(-10, 20, size=(5, 3)).astype(np.int32)
    weights = np.random.uniform(0.5, 2.0, size=(5, 3)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 75.0,
        "axis": 0,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, negative axis=-1, minimum (q=0.0), keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float64)
    weights = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": 0.0,
        "axis": -1,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D int64 array, maximum (q=100.0), keepdims=False
    a = np.array([-50, -20, 0, 30, 100], dtype=np.int64)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 100.0,
        "axis": 0,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis=1, fractional q=50.5, keepdims=False
    a = np.random.randn(3, 8, 4).astype(np.float32)
    weights = np.ones((3, 8, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 50.5,
        "axis": 1,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, axis=2, q=10.0, keepdims=True
    a = np.random.randn(2, 3, 5, 2).astype(np.float32)
    weights = np.random.uniform(1.0, 2.0, size=(2, 3, 5, 2)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 10.0,
        "axis": 2,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array, negative axis=-2, q=99.9, keepdims=False
    a = np.random.uniform(-100.0, -1.0, size=(7, 4)).astype(np.float32)
    weights = np.random.uniform(0.1, 10.0, size=(7, 4)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 99.9,
        "axis": -2,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float64 array, q=33.33, keepdims=True
    a = np.array([10.5, -2.3, 4.8, 100.1, -50.0, 0.0], dtype=np.float64)
    weights = np.array([0.5, 1.5, 2.0, 0.1, 3.0, 1.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": 33.33,
        "axis": 0,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, axis=0, q=80.0, keepdims=False
    a = np.random.randn(6, 4, 2).astype(np.float32)
    weights = np.random.uniform(0.2, 1.5, size=(6, 4, 2)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 80.0,
        "axis": 0,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.percentile_2"] = percentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.percentile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.percentile_2'.")


check_valid('jax.numpy.percentile', generated_inputs['jax.numpy.percentile_2'], lib="jax", suffix=2)
