
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polygamma_inputs():
    list_of_inputs = []

    # Input 1: m=0.0 (digamma), 1D array float32
    m = 0.0
    x = np.array([0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 2: m=1.0 (trigamma), 2D array float32
    m = 1.0
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 3: m=2.0 (tetragamma), 1D array float64
    m = 2.0
    x = np.linspace(0.1, 5.0, 10, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 4: m=0.0, 0D scalar array float32
    m = 0.0
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 5: m=3.0, 3D array float64
    m = 3.0
    x = np.arange(1, 9, dtype=np.float64).reshape((2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 6: m=1.0, 4D array float32
    m = 1.0
    x = np.ones((2, 1, 3, 2), dtype=np.float32) * 1.5
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 7: m=4.0, 1D array with larger values float32
    m = 4.0
    x = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 8: m=5.0, 2D array float64
    m = 5.0
    x = np.array([[0.25, 0.75, 1.25], [1.75, 2.25, 2.75]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 9: m=0.0, 1D array with non-integer negative numbers
    m = 0.0
    x = np.array([-0.5, -1.5, -2.5, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 10: m=2.0, 3D array float32
    m = 2.0
    x = np.full((3, 3, 3), 4.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 11: m=6.0, 1D array float64
    m = 6.0
    x = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    return list_of_inputs


generated_inputs["jax.lax.polygamma_1"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.polygamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.polygamma_1'.")


check_valid('jax.lax.polygamma', generated_inputs['jax.lax.polygamma_1'], lib="jax", suffix=1)
