
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, custom spacing coordinates
    f = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float32)
    varargs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, constant spacing scalar
    f = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    varargs = np.array(2.0, dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, axis 0, constant spacing
    f = np.random.randn(4, 4).astype(np.float32)
    varargs = np.array(1.0, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, axis 1, constant spacing
    f = np.random.randn(3, 5).astype(np.float32)
    varargs = np.array(0.5, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 1,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32 array with negative values, custom spacing, negative axis
    f = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0], dtype=np.float32)
    varargs = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": -1,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis 2, constant spacing
    f = np.random.randn(2, 3, 4).astype(np.float32)
    varargs = np.array(1.5, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 2,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array, negative axis, constant spacing
    f = np.random.randn(5, 5).astype(np.float64)
    varargs = np.array(0.1, dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": -2,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int32 array, custom spacing
    f = np.array([1, 4, 9, 16, 25, 36], dtype=np.int32)
    varargs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small 1D float32 array, constant spacing
    f = np.array([100.0, 50.0, 25.0], dtype=np.float32)
    varargs = np.array(10.0, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float64 array, axis 1, constant spacing
    f = np.random.randn(3, 3, 3).astype(np.float64)
    varargs = np.array(2.5, dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 1,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.gradient_1"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gradient_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gradient_1'.")


check_valid('jax.numpy.gradient', generated_inputs['jax.numpy.gradient_1'], lib="jax", suffix=1)
