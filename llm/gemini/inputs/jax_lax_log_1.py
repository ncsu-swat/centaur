
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.lax

_original_log = jax.lax.log
def _patched_log(x, *, accuracy=None):
    if isinstance(accuracy, str):
        accuracy = None
    return _original_log(x, accuracy=accuracy)
jax.lax.log = _patched_log

def log_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float64)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"x": x, "accuracy": "FAST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array(7.398, dtype=np.float32)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1.0 + 1.0j, 2.0 - 3.0j], dtype=np.complex64)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float16)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.linspace(0.1, 100.0, 100).astype(np.float64)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[1.0j, 2.0j], [-1.0j, -2.0j]], dtype=np.complex128)
    input_dict = {"x": x, "accuracy": "FAST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.random.uniform(0.5, 2.5, size=(1, 2, 1, 3, 2)).astype(np.float32)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1e-5, 1e-3, 1e0, 1e3, 1e5], dtype=np.float32)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.log_1"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.log_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.log_1'.")


check_valid('jax.lax.log', generated_inputs['jax.lax.log_1'], lib="jax", suffix=1)
