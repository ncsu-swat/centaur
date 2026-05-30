
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def average_inputs():
    list_of_inputs = []
    
    # Input 1: 1D float32 array, axis 0
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    weights = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    returned = False
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D float32 array, axis 0, returned=True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    returned = True
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D float32 array, axis 1, keepdims=True, negative values
    a = np.random.uniform(-10, 10, (2, 3)).astype(np.float32)
    axis = 1
    weights = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    returned = False
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, axis 2, keepdims=True, returned=True
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = 2
    weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    returned = True
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, negative axis
    a = np.random.randn(4, 3, 2).astype(np.float32)
    axis = -1
    weights = np.array([1.0, 2.0], dtype=np.float32)
    returned = False
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array
    a = np.random.randn(2, 2, 2, 3).astype(np.float32)
    axis = 3
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    returned = True
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array, integer weights
    a = np.array([10, -20, 30, 40], dtype=np.int32)
    axis = 0
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
    returned = False
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array, weights matching exactly the shape of 'a'
    a = np.random.randn(3, 3).astype(np.float32)
    axis = 1
    weights = np.random.uniform(0.1, 1.0, (3, 3)).astype(np.float32)
    returned = False
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, axis 0, keepdims=True, returned=True
    a = np.random.randn(5, 2, 3).astype(np.float32)
    axis = 0
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    returned = True
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array, axis -2
    a = np.random.randn(3, 4).astype(np.float32)
    axis = -2
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    returned = False
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": returned,
        "keepdims": keepdims
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
