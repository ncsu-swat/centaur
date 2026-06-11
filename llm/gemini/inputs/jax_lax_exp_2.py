
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax._src.lax.lax

_orig_exp = jax.lax.exp
def patched_exp(x, *, accuracy=None):
    return _orig_exp(x, accuracy=None)

jax.lax.exp = patched_exp
jax._src.lax.lax.exp = patched_exp

def exp_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([0.0, 1.0, 2.0, -1.0]).astype(np.float32)
    accuracy = (1e-5, 1e-5)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.random.randn(3, 3).astype(np.float64)
    accuracy = (1e-6, 1e-6)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = -np.abs(np.random.randn(2, 2, 2).astype(np.float32))
    accuracy = (1e-4, 1e-4)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array(1.5).astype(np.float32)
    accuracy = (1e-5, 1e-5)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.random.randn(2, 2, 2, 2).astype(np.float16)
    accuracy = (1e-3, 1e-3)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    accuracy = (1e-5, 1e-5)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex128)
    accuracy = (1e-6, 1e-6)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.zeros((5,), dtype=np.float32)
    accuracy = (1e-5, 1e-5)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.linspace(-5.0, 5.0, 12).reshape(3, 4).astype(np.float32)
    accuracy = (1e-5, 1e-5)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.random.uniform(-10, 10, (2, 3, 4)).astype(np.float64)
    accuracy = (1e-7, 1e-7)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.exp_2"] = exp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.exp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.exp_2'.")


check_valid('jax.lax.exp', generated_inputs['jax.lax.exp_2'], lib="jax", suffix=2)
