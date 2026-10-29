
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def round_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, decimals=0
    a = np.array([1.532, 3.267, 6.149, -2.5, 2.5], dtype=np.float32)
    decimals = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 2: 1D float32 array, decimals=2
    a = np.array([1.5326, 3.2671, 6.1499, -0.005], dtype=np.float32)
    decimals = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 3: 2D float64 array, decimals=1
    a = np.array([[-10.55, 21.45], [12.55, -31.55]], dtype=np.float64)
    decimals = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 4: 3D float32 array, decimals=3
    a = np.random.randn(2, 3, 4).astype(np.float32)
    decimals = 3
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 5: 0D scalar array, decimals=0
    a = np.array(4.5, dtype=np.float32)
    decimals = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 6: Integer array (int32), decimals=0
    a = np.array([-5, 0, 3, 10], dtype=np.int32)
    decimals = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 7: Float16 2D array, decimals=1
    a = np.array([[0.125, -0.875], [1.5, -2.5]], dtype=np.float16)
    decimals = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 8: 4D float32 array, decimals=4
    a = np.random.uniform(-10.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    decimals = 4
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 9: 1D float64 array with zeros and infinities, decimals=2
    a = np.array([0.0, -0.0, 1e-5, -1e-5], dtype=np.float64)
    decimals = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 10: 2D integer array (int64), decimals=2
    a = np.array([[10, 20], [30, 40]], dtype=np.int64)
    decimals = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

    # Input 11: 1D complex64 array, decimals=1
    a = np.array([1.23 + 4.56j, -2.34 - 5.67j], dtype=np.complex64)
    decimals = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "decimals": decimals}))

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
