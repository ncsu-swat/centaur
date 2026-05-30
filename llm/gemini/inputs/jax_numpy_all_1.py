
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def all_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, axis 0, keepdims False
    list_of_inputs.append({
        "a": np.array([True, False, True, True, False], dtype=bool),
        "axis": 0,
        "keepdims": False,
        "where": np.array([True, True, False, True, False], dtype=bool)
    })

    # Input 2: 2D integer array, axis 1, keepdims True
    list_of_inputs.append({
        "a": np.array([[1, 2], [0, 4]], dtype=np.int32),
        "axis": 1,
        "keepdims": True,
        "where": np.array([[True, True], [True, False]], dtype=bool)
    })

    # Input 3: 3D boolean array, axis 2, keepdims False, randomized values
    list_of_inputs.append({
        "a": (np.random.rand(3, 4, 5) > 0.5),
        "axis": 2,
        "keepdims": False,
        "where": (np.random.rand(3, 4, 5) > 0.2)
    })

    # Input 4: 1D float32 array with negative and zero values, negative axis
    list_of_inputs.append({
        "a": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "axis": -1,
        "keepdims": True,
        "where": np.array([True, True, False, True], dtype=bool)
    })

    # Input 5: 3D all-True boolean array, axis 1, keepdims False
    list_of_inputs.append({
        "a": np.ones((2, 2, 2), dtype=bool),
        "axis": 1,
        "keepdims": False,
        "where": np.ones((2, 2, 2), dtype=bool)
    })

    # Input 6: 2D boolean array, axis -2, keepdims True
    list_of_inputs.append({
        "a": np.array([[False, True], [True, False]], dtype=bool),
        "axis": -2,
        "keepdims": True,
        "where": np.array([[True, True], [False, False]], dtype=bool)
    })

    # Input 7: 2D int64 array of zeros, axis 0, keepdims False
    list_of_inputs.append({
        "a": np.zeros((5, 1), dtype=np.int64),
        "axis": 0,
        "keepdims": False,
        "where": np.ones((5, 1), dtype=bool)
    })

    # Input 8: 3D randomized boolean array, axis 2, keepdims True
    list_of_inputs.append({
        "a": (np.random.randint(0, 2, size=(3, 3, 3)) == 1),
        "axis": 2,
        "keepdims": True,
        "where": (np.random.randint(0, 2, size=(3, 3, 3)) == 1)
    })

    # Input 9: 1D float64 array, axis 0, keepdims False
    list_of_inputs.append({
        "a": np.array([1.5, -2.3, 0.0], dtype=np.float64),
        "axis": 0,
        "keepdims": False,
        "where": np.array([True, True, True], dtype=bool)
    })

    # Input 10: 2D boolean array, broadcastable 'where' mask, axis 0, keepdims True
    list_of_inputs.append({
        "a": np.array([[True, True], [True, True]], dtype=bool),
        "axis": 0,
        "keepdims": True,
        "where": np.array([[True, False]], dtype=bool)
    })

    return [copy.deepcopy(x) for x in list_of_inputs]

generated_inputs["jax.numpy.all_1"] = all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.all_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.all_1'.")


check_valid('jax.numpy.all', generated_inputs['jax.numpy.all_1'], lib="jax", suffix=1)
