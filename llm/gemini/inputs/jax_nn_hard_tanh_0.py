
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hard_tanh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with values spanning across and beyond [-1, 1]
    x = np.array([-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float32 array with random normal values
    x = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar tensor float32
    x = np.array(0.75, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar tensor negative beyond bounds
    x = np.array(-3.14, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float64 array
    x = np.random.uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D float32 array simulating a batch of feature maps
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D float16 array
    x = np.linspace(-2.0, 2.0, 15, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D integer array (jax converts/handles integers properly)
    x = np.array([[-3, -1, 0], [1, 2, 5]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array containing exact boundary values and zeros
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 5D float32 array
    x = np.random.randn(1, 2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D int64 array with extreme values
    x = np.array([-100, -1, 0, 1, 100], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.hard_tanh"] = hard_tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.hard_tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.hard_tanh'.")


check_valid('jax.nn.hard_tanh', generated_inputs['jax.nn.hard_tanh'], lib="jax", suffix=0)
