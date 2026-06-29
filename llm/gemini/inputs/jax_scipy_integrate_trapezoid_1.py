
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    input_dict = {
        "y": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "dx": 1.0,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64
    input_dict = {
        "y": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        "x": np.array([0.0, 0.5, 1.0], dtype=np.float64),
        "dx": 0.5,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32, axis 0
    input_dict = {
        "y": np.arange(6, dtype=np.float32).reshape(2, 3),
        "x": np.array([0.0, 1.0], dtype=np.float32),
        "dx": 1.0,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32, axis 1
    input_dict = {
        "y": np.arange(6, dtype=np.float32).reshape(2, 3),
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "dx": 2.0,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64, axis -1
    input_dict = {
        "y": np.arange(4, dtype=np.float64).reshape(2, 2),
        "x": np.array([0.0, 0.1], dtype=np.float64),
        "dx": 0.1,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32, axis 1
    input_dict = {
        "y": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        "x": np.array([0.0, 1.0], dtype=np.float32),
        "dx": 1.0,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32, axis 2
    input_dict = {
        "y": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        "x": np.array([1.0, 2.0, 4.0], dtype=np.float32),
        "dx": 0.5,
        "axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64, axis 0
    input_dict = {
        "y": np.arange(12, dtype=np.float64).reshape(3, 2, 2),
        "x": np.array([0.0, 1.5, 3.0], dtype=np.float64),
        "dx": 1.5,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 constant
    input_dict = {
        "y": np.ones((5,), dtype=np.float32),
        "x": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "dx": 1.0,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32
    input_dict = {
        "y": np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2),
        "x": np.array([0.0, 1.0], dtype=np.float32),
        "dx": 1.0,
        "axis": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.integrate.trapezoid_1"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.integrate.trapezoid_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.integrate.trapezoid_1'.")


check_valid('jax.scipy.integrate.trapezoid', generated_inputs['jax.scipy.integrate.trapezoid_1'], lib="jax", suffix=1)
