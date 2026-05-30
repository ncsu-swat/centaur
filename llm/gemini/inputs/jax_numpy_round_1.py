
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def round_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, decimals=0
    a = np.array([1.2, 2.7, 3.5, 4.9], dtype=np.float32)
    input_dict = {"a": a, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array with negative values, decimals=0
    a = np.array([-1.2, -2.7, -3.5, -4.9], dtype=np.float32)
    input_dict = {"a": a, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Halfway values (round to even), decimals=0
    a = np.array([10.5, 21.5, 12.5, 31.5], dtype=np.float32)
    input_dict = {"a": a, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, decimals=1
    a = np.array([[1.53, 2.57], [3.11, 4.89]], dtype=np.float32)
    input_dict = {"a": a, "decimals": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 array, decimals=2
    a = np.array([[1.532, 2.578], [3.111, 4.899]], dtype=np.float64)
    input_dict = {"a": a, "decimals": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, decimals=3
    a = np.random.uniform(low=-10.0, high=10.0, size=(2, 3, 3)).astype(np.float32)
    input_dict = {"a": a, "decimals": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Halfway values with decimals=1
    a = np.array([1.25, 1.35, 2.45, 2.55], dtype=np.float64)
    input_dict = {"a": a, "decimals": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D array (scalar equivalent), decimals=0
    a = np.array(5.67, dtype=np.float32)
    input_dict = {"a": a, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Integer array (should return same values), decimals=0
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"a": a, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array with large number of decimals, decimals=5
    a = np.random.uniform(low=0.0, high=1.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"a": a, "decimals": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D float64 array, decimals=4
    a = np.array([0.123456, 9.876543, -4.56789], dtype=np.float64)
    input_dict = {"a": a, "decimals": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.round_1"] = round_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.round_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.round_1'.")


check_valid('jax.numpy.round', generated_inputs['jax.numpy.round_1'], lib="jax", suffix=1)
