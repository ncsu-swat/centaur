
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def acos_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array float32 with values between -1 and 1
    x = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array float64
    x = np.array([[-0.9, -0.5, 0.0], [0.1, 0.5, 0.9]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array float16
    x = np.array([-1.0, -0.707, 0.0, 0.707, 1.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array float32
    x = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array float64
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 3, 3)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array complex64
    x = np.array([-0.5 + 0.5j, 0.2 - 0.3j, 1.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D array complex128
    x = np.array(
        [[1.5 + 2.0j, -1.0 - 0.5j], [0.0 + 1.0j, -0.5 + 0.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Boundary values float32
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array float32
    x = np.array([], dtype=np.float32).reshape(0, 5)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.acos"] = acos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.acos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.acos'.")


check_valid('jax.lax.acos', generated_inputs['jax.lax.acos'], lib="jax", suffix=0)
