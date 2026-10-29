
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def average_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, matching shape weights, axis=0
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "weights": weights,
        "returned": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, 1D weights matching axis 1, returned=True
    a = np.random.randn(3, 4).astype(np.float32)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 1,
        "weights": weights,
        "returned": True,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, weights matching full shape, keepdims=True
    a = np.random.randn(4, 5).astype(np.float64)
    weights = np.random.uniform(0.5, 2.0, size=(4, 5)).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 0,
        "weights": weights,
        "returned": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array, 1D weights along axis 2, returned=True, keepdims=True
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 2,
        "weights": weights,
        "returned": True,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, negative axis=-1, 1D weights
    a = np.random.randn(3, 2, 5).astype(np.float32)
    weights = np.ones(5, dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": -1,
        "weights": weights,
        "returned": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array, negative axis=-1, returned=True, keepdims=True
    a = np.array([-5, -2, 0, 3, 7], dtype=np.int64)
    weights = np.array([2, 1, 1, 3, 2], dtype=np.int64)
    input_dict = {
        "a": a,
        "axis": -1,
        "weights": weights,
        "returned": True,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, weights matching full shape, axis=1
    a = np.random.randn(2, 3, 4, 2).astype(np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(2, 3, 4, 2)).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 1,
        "weights": weights,
        "returned": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D complex64 array, 1D weights matching axis 0
    a = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(np.complex64)
    weights = np.array([1.0, 2.0, 1.5, 0.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "weights": weights,
        "returned": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 array with full weights, returned=True, keepdims=False
    a = np.random.randn(2, 4, 3).astype(np.float64)
    weights = np.random.exponential(1.0, size=(2, 4, 3)).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 1,
        "weights": weights,
        "returned": True,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array, negative axis=-2, 1D weights matching axis -2
    a = np.random.randn(5, 3).astype(np.float32)
    weights = np.array([1.0, 3.0, 2.0, 4.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": -2,
        "weights": weights,
        "returned": True,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.average_1"] = average_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.average_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.average_1'.")


check_valid('jax.numpy.average', generated_inputs['jax.numpy.average_1'], lib="jax", suffix=1)
