
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default step 1.0, axis (0,)
    f = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": (0,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with negative values and custom spacing
    f = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": 0.5,
        "axis": (0,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array along axis 0
    f = np.arange(12, dtype=np.float32).reshape(3, 4)
    input_dict = {
        "f": f,
        "varargs": 2.0,
        "axis": (0,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array along axis 1
    f = np.random.randn(5, 6).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.5,
        "axis": (1,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array along both axes (0, 1)
    f = np.random.uniform(-5.0, 5.0, size=(4, 5)).astype(np.float64)
    input_dict = {
        "f": f,
        "varargs": 0.1,
        "axis": (0, 1),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array using negative axis index (-1,)
    f = np.cos(np.linspace(0, np.pi, 20)).reshape(4, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 3.0,
        "axis": (-1,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array along axes (0, 2)
    f = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.25,
        "axis": (0, 2),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array along all axes (0, 1, 2)
    f = np.ones((4, 4, 4), dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": (0, 1, 2),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array using negative axis index (-2,)
    f = np.sin(np.arange(24, dtype=np.float32)).reshape(2, 3, 4)
    input_dict = {
        "f": f,
        "varargs": 2.5,
        "axis": (-2,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array along subset of axes (1, 3)
    f = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.5,
        "axis": (1, 3),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.gradient_5"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gradient_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gradient_5'.")


check_valid('jax.numpy.gradient', generated_inputs['jax.numpy.gradient_5'], lib="jax", suffix=5)
