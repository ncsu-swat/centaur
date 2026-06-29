
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_sign_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D
    x = np.array([-2.5, -0.0, 0.0, 3.14], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D
    x = np.array([[-10.0, 20.0], [0.0, -5.0]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, 1D
    x = np.array([-10, 0, 15], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int16, 3D
    x = np.array([[[-1, 2], [0, -3]], [[4, -5], [6, 0]]], dtype=np.int16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D
    x = np.array([3 + 4j, -1 - 1j, 0 + 0j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 with NaN, inf, -inf, -0.0
    x = np.array([np.nan, np.inf, -np.inf, -0.0, +0.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, 4D
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int8, 2D
    x = np.array([[-128, 0, 127], [-5, 5, 0]], dtype=np.int8)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128, 2D
    x = np.array([[1j, -1j], [1 + 1j, -1 - 1j]], dtype=np.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 scalar (0D array)
    x = np.array(-1.5, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int64, 1D
    x = np.array([-9223372036854775808, 0, 9223372036854775807], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.sign"] = jax_lax_sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sign'.")


check_valid('jax.lax.sign', generated_inputs['jax.lax.sign'], lib="jax", suffix=0)
