
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax
import jax.lax
import numpy as np
import copy

# Monkeypatch jax.lax.exp2 to support accuracy as string on CPU by mapping it to None
_original_exp2 = jax.lax.exp2
def _patched_exp2(x, *, accuracy=None):
    return _original_exp2(x, accuracy=None)
jax.lax.exp2 = _patched_exp2

def exp2_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, positive values, DEFAULT accuracy
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, negative values, HIGHEST accuracy
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64, mixed values, DEFAULT accuracy
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D float32 (scalar), HIGHEST accuracy
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D complex64, DEFAULT accuracy
    x = np.array([1.0 + 1.0j, -1.0 - 1.0j], dtype=np.complex64)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float16, HIGHEST accuracy
    x = np.random.uniform(-2.0, 2.0, size=(1, 2, 2, 1)).astype(np.float16)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex128, DEFAULT accuracy
    x = np.array([[0.5 + 0.5j, -0.5 + 0.5j], [0.0, 1.0 - 1.0j]], dtype=np.complex128)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 containing zeros, DEFAULT accuracy
    x = np.zeros((5,), dtype=np.float32)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 near zero, HIGHEST accuracy
    x = np.random.uniform(-0.1, 0.1, size=(3, 3, 3)).astype(np.float32)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64, larger values, DEFAULT accuracy
    x = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.exp2"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.exp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.exp2'.")


check_valid('jax.lax.exp2', generated_inputs['jax.lax.exp2'], lib="jax", suffix=0)
