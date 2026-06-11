
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_exp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, accuracy None
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array, accuracy None
    x = np.random.randn(3, 3).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float16 array, accuracy None
    x = np.random.uniform(-5.0, 5.0, (2, 2, 2)).astype(np.float16)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar (0D array) float32, accuracy None
    x = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 1D array, accuracy None
    x = np.array([1.0 + 1j, -2.0 - 2j], dtype=np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex128 2D array, accuracy None
    x = (np.random.randn(2, 4) + 1j * np.random.randn(2, 4)).astype(np.complex128)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger float32 4D array, accuracy None
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values float64, accuracy None
    x = np.array([-10.0, -100.0, -0.5], dtype=np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive values float32, accuracy None
    x = np.array([10.0, 20.0, 50.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array with infinity and nan, accuracy None
    x = np.array([-np.inf, np.inf, np.nan], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.exp_1"] = jax_lax_exp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.exp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.exp_1'.")


check_valid('jax.lax.exp', generated_inputs['jax.lax.exp_1'], lib="jax", suffix=1)
