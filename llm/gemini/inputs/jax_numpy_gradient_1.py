
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D array with uniform coordinates
    f = np.linspace(-5.0, 5.0, 10, dtype=np.float32)
    varargs = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 0,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with non-uniform coordinates and negative axis
    f = np.array(
        [-2.5, -1.0, 0.0, 1.5, 3.0, 2.0, 0.5, -1.2], dtype=np.float64
    )
    varargs = np.array(
        [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8], dtype=np.float64
    )
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": -1,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, gradient along axis 0
    f = np.random.randn(5, 6).astype(np.float32)
    varargs = np.linspace(0.0, 2.0, 5, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 0,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, gradient along axis 1
    f = np.random.randn(4, 7).astype(np.float64)
    varargs = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 1,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, gradient along negative axis -1
    f = np.random.uniform(-10.0, 10.0, size=(6, 8)).astype(np.float32)
    varargs = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": -1,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, gradient along axis 0
    f = np.random.randn(3, 4, 5).astype(np.float32)
    varargs = np.arange(3, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 0,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, gradient along axis 1
    f = np.random.randn(4, 5, 6).astype(np.float64)
    varargs = np.linspace(0.0, 10.0, 5, dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 1,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, gradient along axis 2
    f = np.random.randn(2, 3, 4).astype(np.float32)
    varargs = np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 2,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, gradient along axis 1
    f = np.random.randn(2, 3, 4, 5).astype(np.float32)
    varargs = np.array([1.0, 2.5, 4.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 1,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, gradient along negative axis -2
    f = np.random.randn(5, 5).astype(np.float64)
    varargs = np.linspace(0.0, 4.0, 5, dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": -2,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array with decreasing coordinates
    f = np.sin(np.linspace(0.0, np.pi, 12)).astype(np.float32)
    varargs = np.linspace(10.0, 0.0, 12, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": varargs,
        "axis": 0,
        "edge_order": None,
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
