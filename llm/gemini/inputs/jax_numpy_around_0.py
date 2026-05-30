
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def around_inputs():
    list_of_inputs = []

    a = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    decimals = 0
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

    a = np.array([[1.23, 2.37, 3.56], [4.12, 5.67, 6.89]], dtype=np.float64)
    decimals = 1
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

    a = np.array([[[12.3, 25.7], [34.1, 48.9]], [[51.2, 67.8], [73.4, 89.1]]], dtype=np.float32)
    decimals = -1
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

    a = np.array([0.1234, 1.5678, -2.3456], dtype=np.float16)
    decimals = 2
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

    a = np.array(3.14159, dtype=np.float32)
    decimals = 0
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

    a = np.random.uniform(100, 1000, size=(2, 2, 2, 2)).astype(np.float64)
    decimals = -2
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

    a = np.random.uniform(-10, 10, size=(3, 3)).astype(np.float32)
    decimals = 3
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

    a = np.array([0.000123, -0.000456, 0.000789], dtype=np.float64)
    decimals = 4
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

    a = np.array([[1234.5, 6789.0], [10500.2, 98765.4]], dtype=np.float32)
    decimals = -3
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

    a = np.array([[-1.15, -2.25], [3.35, 4.45]], dtype=np.float16)
    decimals = 1
    list_of_inputs.append({"a": copy.deepcopy(a), "decimals": decimals})

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
