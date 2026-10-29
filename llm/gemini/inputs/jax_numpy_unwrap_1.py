
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unwrap_inputs():
    list_of_inputs = []

    # Input 1: 1D array, standard radians
    p = np.array([0.0, 1.5, 3.0, -3.0, -1.5, 0.0], dtype=np.float32)
    input_dict = {
        "p": p,
        "discont": float(np.pi),
        "axis": 0,
        "period": float(2 * np.pi),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, degree scale
    p = np.array([350.0, 5.0, 20.0, 355.0, 10.0], dtype=np.float64)
    input_dict = {
        "p": p,
        "discont": 180.0,
        "axis": -1,
        "period": 360.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array along axis 0
    p = np.array(
        [[0.0, 1.0], [3.0, 4.0], [-3.0, -2.0], [0.5, 1.5]], dtype=np.float32
    )
    input_dict = {
        "p": p,
        "discont": float(np.pi),
        "axis": 0,
        "period": float(2 * np.pi),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array along axis 1, custom period
    rng = np.random.default_rng(42)
    p = (rng.uniform(-50.0, 50.0, size=(4, 8))).astype(np.float64)
    input_dict = {
        "p": p,
        "discont": 50.0,
        "axis": 1,
        "period": 100.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array along axis -1
    p = rng.uniform(-np.pi, np.pi, size=(2, 3, 5)).astype(np.float32)
    input_dict = {
        "p": p,
        "discont": 3.14,
        "axis": -1,
        "period": float(2 * np.pi),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array along axis 1, normalized [0, 1) period
    p = rng.uniform(0.0, 1.0, size=(3, 6, 2)).astype(np.float64)
    input_dict = {
        "p": p,
        "discont": 0.5,
        "axis": 1,
        "period": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with negative numbers and custom discont
    p = np.array([-180.0, -90.0, 0.0, 90.0, 180.0, -170.0], dtype=np.float32)
    input_dict = {
        "p": p,
        "discont": 90.0,
        "axis": 0,
        "period": 360.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array along axis 2
    p = rng.uniform(-np.pi, np.pi, size=(2, 2, 8, 2)).astype(np.float64)
    input_dict = {
        "p": p,
        "discont": float(np.pi),
        "axis": 2,
        "period": float(2 * np.pi),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array along negative axis -2
    p = rng.uniform(0.0, 360.0, size=(10, 4)).astype(np.float32)
    input_dict = {
        "p": p,
        "discont": 120.0,
        "axis": -2,
        "period": 360.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with period 4.0
    p = np.array([0.5, 1.5, 2.5, 3.5, 0.2, 1.2], dtype=np.float64)
    input_dict = {
        "p": p,
        "discont": 2.0,
        "axis": -1,
        "period": 4.0,
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
