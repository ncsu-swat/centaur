
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_ptp_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, axis=(0,), keepdims=False
    a = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32, negative/positive, axis=(1,), keepdims=True
    a = np.random.uniform(-10, 10, size=(5, 5)).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (1,),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, int32, axis=(0, 2), keepdims=False
    a = np.random.randint(-50, 50, size=(2, 3, 4)).astype(np.int32)
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, float64, axis=(0,), keepdims=True
    a = np.random.randn(10).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": (0,),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, int16, axis=(1, 3), keepdims=False
    a = np.random.randint(-100, 100, size=(2, 3, 2, 4)).astype(np.int16)
    input_dict = {
        "a": a,
        "axis": (1, 3),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, float32, axis=(0, 1), keepdims=False
    a = np.random.randn(10, 10).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, uint8, axis=(1,), keepdims=True
    a = np.random.randint(0, 255, size=(3, 5, 3)).astype(np.uint8)
    input_dict = {
        "a": a,
        "axis": (1,),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, float64, axis=(0, 1), keepdims=True
    a = np.random.randn(4, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, float32, axis=(2, 4), keepdims=False
    a = np.random.randn(2, 2, 3, 2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (2, 4),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, int32, axis=(2,), keepdims=True
    a = np.random.randint(-10, 10, size=(4, 4, 4)).astype(np.int32)
    input_dict = {
        "a": a,
        "axis": (2,),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ptp_2"] = jax_numpy_ptp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ptp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ptp_2'.")


check_valid('jax.numpy.ptp', generated_inputs['jax.numpy.ptp_2'], lib="jax", suffix=2)
