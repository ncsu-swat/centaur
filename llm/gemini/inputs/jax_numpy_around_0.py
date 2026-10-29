
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def around_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, decimals=0
    a = np.array([0.1, 0.5, 0.9, -1.2, -1.7], dtype=np.float32)
    decimals = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 2: 2D float32 array, decimals=2
    a = np.array([[1.2345, 2.3456], [-3.4567, -4.5678]], dtype=np.float32)
    decimals = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 3: 1D float64 array, negative decimals
    a = np.array([123.45, 678.9, 12.0, 999.0], dtype=np.float64)
    decimals = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 4: 0D scalar array, decimals=1
    a = np.array(3.14159, dtype=np.float32)
    decimals = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 5: 3D float32 array, decimals=3
    a = np.random.randn(2, 3, 4).astype(np.float32)
    decimals = 3
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 6: 2D float32 array with negative decimals
    a = np.array([[120.0, 450.0], [980.0, 1550.0]], dtype=np.float32)
    decimals = -2
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 7: 1D float16 array, decimals=1
    a = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float16)
    decimals = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 8: 4D float64 array, decimals=4
    a = np.random.uniform(-10, 10, size=(2, 2, 3, 2)).astype(np.float64)
    decimals = 4
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 9: Large negative decimals
    a = np.array([123456.0, 789101.0, -987654.0], dtype=np.float32)
    decimals = -4
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 10: 1D float64 array with zeros and small decimals
    a = np.array([0.0, -0.0, 1.0001, -1.0001], dtype=np.float64)
    decimals = 3
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 11: 2D integer array with non-negative decimals
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    decimals = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    return list_of_inputs


generated_inputs["jax.numpy.around"] = around_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.around' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.around'.")


check_valid('jax.numpy.around', generated_inputs['jax.numpy.around'], lib="jax", suffix=0)
