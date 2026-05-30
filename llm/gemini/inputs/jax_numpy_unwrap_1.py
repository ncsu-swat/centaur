
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unwrap_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, standard 2*pi period
    p = np.array([0.1, 0.5, 1.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "p": p,
        "discont": 3.141592653589793,
        "axis": -1,
        "period": 6.283185307179586
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64, 360 degree period
    p = np.array([0.0, 90.0, 180.0, 270.0, 350.0, 10.0], dtype=np.float64)
    input_dict = {
        "p": p,
        "discont": 180.0,
        "axis": 0,
        "period": 360.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32, axis 0, standard 2*pi period
    p = np.array([[0.0, 1.0], [5.0, 6.0]], dtype=np.float32)
    input_dict = {
        "p": p,
        "discont": 3.141592653589793,
        "axis": 0,
        "period": 6.283185307179586
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32, axis 1, custom discont
    p = np.array([[10.0, 190.0, 350.0], [20.0, 200.0, 340.0]], dtype=np.float32)
    input_dict = {
        "p": p,
        "discont": 100.0,
        "axis": 1,
        "period": 360.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64, axis 2
    p = np.random.uniform(-3.14, 3.14, size=(2, 3, 4)).astype(np.float64)
    input_dict = {
        "p": p,
        "discont": 3.14,
        "axis": 2,
        "period": 6.28
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float32, period 2.0
    p = np.array([-0.9, -0.5, 0.0, 0.5, 0.9, -0.9], dtype=np.float32)
    input_dict = {
        "p": p,
        "discont": 1.0,
        "axis": -1,
        "period": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32, period 10.0
    p = np.array([[1.0, 9.0], [2.0, 8.0]], dtype=np.float32)
    input_dict = {
        "p": p,
        "discont": 5.0,
        "axis": -1,
        "period": 10.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32, axis 0, period 360.0
    p = np.random.uniform(0.0, 360.0, size=(3, 2, 2)).astype(np.float32)
    input_dict = {
        "p": p,
        "discont": 180.0,
        "axis": 0,
        "period": 360.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float64, period 1.0
    p = np.array([0.1, 0.2, 0.9, 0.1, 0.2], dtype=np.float64)
    input_dict = {
        "p": p,
        "discont": 0.5,
        "axis": 0,
        "period": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64, axis -2
    p = np.random.uniform(-3.14, 3.14, size=(4, 5)).astype(np.float64)
    input_dict = {
        "p": p,
        "discont": 2.0,
        "axis": -2,
        "period": 6.28
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unwrap_1"] = unwrap_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unwrap_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unwrap_1'.")


check_valid('jax.numpy.unwrap', generated_inputs['jax.numpy.unwrap_1'], lib="jax", suffix=1)
