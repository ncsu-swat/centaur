
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rsqrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array
    x = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array
    x = np.random.uniform(1.0, 100.0, size=(2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float32 array
    x = np.random.uniform(0.5, 2.0, size=(2, 3, 2, 1)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float16 array
    x = np.array([0.25, 0.0625, 100.0], dtype=np.float16)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex64 array
    x = (np.random.uniform(1.0, 5.0, size=(2, 2)) + 1j * np.random.uniform(1.0, 5.0, size=(2, 2))).astype(np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D float32 array
    x = np.array(4.0, dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D float32 array
    x = np.random.uniform(0.1, 1.0, size=(1, 2, 1, 3, 2)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array
    x = np.random.uniform(10.0, 20.0, size=(4, 2)).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D complex128 array
    x = (np.random.uniform(0.5, 2.0, size=(2, 2, 2)) + 1j * np.random.uniform(0.5, 2.0, size=(2, 2, 2))).astype(np.complex128)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.rsqrt_1"] = rsqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.rsqrt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.rsqrt_1'.")


check_valid('jax.lax.rsqrt', generated_inputs['jax.lax.rsqrt_1'], lib="jax", suffix=1)
