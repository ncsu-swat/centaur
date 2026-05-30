
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def any_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D boolean array, axis 0, keepdims True, same shape where
    a = np.array([[True, False, True], [False, False, True]], dtype=bool)
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D boolean array, axis 0, keepdims False, same shape where
    a = np.array([True, False, True, False], dtype=bool)
    where = np.array([True, True, True, False], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float array, axis 1, keepdims True, same shape where
    a = np.random.randn(2, 3, 4)
    where = np.random.choice([True, False], size=(2, 3, 4))
    input_dict = {
        "a": a,
        "axis": 1,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D integer array, axis -1, keepdims False, broadcasted where
    a = np.random.randint(-5, 5, size=(4, 5))
    where = np.random.choice([True, False], size=(1, 5))
    input_dict = {
        "a": a,
        "axis": -1,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D boolean array, axis 2, keepdims True, same shape where
    a = np.random.choice([True, False], size=(2, 2, 3, 3))
    where = np.random.choice([True, False], size=(2, 2, 3, 3))
    input_dict = {
        "a": a,
        "axis": 2,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D boolean array (all False), axis 1, keepdims False, same shape where
    a = np.zeros((3, 3), dtype=bool)
    where = np.ones((3, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float array, axis -1, keepdims True, same shape where
    a = np.array([0.0, 1.5, -2.3, 0.0], dtype=np.float32)
    where = np.array([True, True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D integer array, axis 2, keepdims False, broadcasted where
    a = np.ones((2, 2, 2), dtype=np.int32)
    where = np.array([[[True, False]], [[False, True]]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 2,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D boolean array, axis -2, keepdims True, same shape where
    a = np.random.choice([True, False], size=(5, 5))
    where = np.random.choice([True, False], size=(5, 5))
    input_dict = {
        "a": a,
        "axis": -2,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D boolean array, axis 3, keepdims False, broadcasted where
    a = np.random.choice([True, False], size=(2, 1, 3, 2, 2))
    where = np.random.choice([True, False], size=(2, 1, 1, 2, 2))
    input_dict = {
        "a": a,
        "axis": 3,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.any_1"] = any_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.any_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.any_1'.")


check_valid('jax.numpy.any', generated_inputs['jax.numpy.any_1'], lib="jax", suffix=1)
