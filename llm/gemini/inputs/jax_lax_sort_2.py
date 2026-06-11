
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sort_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, sort last dimension, stable=True, num_keys=1
    operand1 = np.array([3.0, 1.0, 2.0, -1.0, 0.0], dtype=np.float32)
    list_of_inputs.append({
        "operand": operand1,
        "dimension": -1,
        "is_stable": True,
        "num_keys": 1
    })

    # Input 2: 2D int32 array, sort along dimension 0, stable=True, num_keys=1
    operand2 = np.array([[5, 4, 3], [1, 2, 6]], dtype=np.int32)
    list_of_inputs.append({
        "operand": operand2,
        "dimension": 0,
        "is_stable": True,
        "num_keys": 1
    })

    # Input 3: 2D float64 array, sort along dimension 1, stable=False, num_keys=1
    operand3 = np.array([[10.5, -1.5], [3.2, 0.0]], dtype=np.float64)
    list_of_inputs.append({
        "operand": operand3,
        "dimension": 1,
        "is_stable": False,
        "num_keys": 1
    })

    # Input 4: 1D int64 array, stable=True, num_keys=1
    operand4 = np.array([-10, 50, 0, -5], dtype=np.int64)
    list_of_inputs.append({
        "operand": operand4,
        "dimension": 0,
        "is_stable": True,
        "num_keys": 1
    })

    # Input 5: 3D float32 array, sort along dimension 2, stable=True, num_keys=1
    operand5 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({
        "operand": operand5,
        "dimension": 2,
        "is_stable": True,
        "num_keys": 1
    })

    # Input 6: 3D int32 array, sort along dimension -1, stable=False, num_keys=1
    operand6 = np.random.randint(-10, 10, size=(2, 2, 3)).astype(np.int32)
    list_of_inputs.append({
        "operand": operand6,
        "dimension": -1,
        "is_stable": False,
        "num_keys": 1
    })

    # Input 7: 4D float32 array, sorting along dimension 1, stable=True, num_keys=1
    operand7 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "operand": operand7,
        "dimension": 1,
        "is_stable": True,
        "num_keys": 1
    })

    # Input 8: 1D float32 array, stable=True, num_keys=1
    operand8 = np.array([1.5, -2.5, 3.0, 0.0], dtype=np.float32)
    list_of_inputs.append({
        "operand": operand8,
        "dimension": 0,
        "is_stable": True,
        "num_keys": 1
    })

    # Input 9: 2D int32 array, stable=False, num_keys=1
    operand9 = np.array([[1, 2], [0, -1]], dtype=np.int32)
    list_of_inputs.append({
        "operand": operand9,
        "dimension": -1,
        "is_stable": False,
        "num_keys": 1
    })

    # Input 10: 1D float64 array, stable=True, num_keys=1
    operand10 = np.array([10.0, 50.0, 0.0, 5.0], dtype=np.float64)
    list_of_inputs.append({
        "operand": operand10,
        "dimension": 0,
        "is_stable": True,
        "num_keys": 1
    })

    return list_of_inputs

generated_inputs["jax.lax.sort_2"] = sort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sort_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sort_2'.")


check_valid('jax.lax.sort', generated_inputs['jax.lax.sort_2'], lib="jax", suffix=2)
