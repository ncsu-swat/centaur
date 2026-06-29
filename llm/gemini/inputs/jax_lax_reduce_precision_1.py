
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reduce_precision_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32 -> bfloat16 equivalent (8 exponent, 7 mantissa)
    operand = np.array([-1.5, 0.0, 1.5, 2.3, 3.14], dtype=np.float32)
    input_dict = {"operand": operand, "exponent_bits": 8, "mantissa_bits": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32 -> float16 equivalent (5 exponent, 10 mantissa)
    operand = np.random.randn(3, 5).astype(np.float32)
    input_dict = {"operand": operand, "exponent_bits": 5, "mantissa_bits": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, float32 -> custom low precision (4 exponent, 3 mantissa)
    operand = np.random.uniform(-10, 10, size=(2, 2, 2)).astype(np.float32)
    input_dict = {"operand": operand, "exponent_bits": 4, "mantissa_bits": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, float64 -> float32 equivalent (8 exponent, 23 mantissa)
    operand = np.array([1e-10, 1e10, -1e10], dtype=np.float64)
    input_dict = {"operand": operand, "exponent_bits": 8, "mantissa_bits": 23}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, float32 -> (6 exponent, 12 mantissa)
    operand = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"operand": operand, "exponent_bits": 6, "mantissa_bits": 12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D array (scalar), float32 -> (5 exponent, 10 mantissa)
    operand = np.array(3.14159, dtype=np.float32)
    input_dict = {"operand": operand, "exponent_bits": 5, "mantissa_bits": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with large values, float64 -> (10 exponent, 20 mantissa)
    operand = np.array([[1e100, -1e100], [0.0, 1.0]], dtype=np.float64)
    input_dict = {"operand": operand, "exponent_bits": 10, "mantissa_bits": 20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, float32 -> extreme low precision (2 exponent, 2 mantissa)
    operand = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    input_dict = {"operand": operand, "exponent_bits": 2, "mantissa_bits": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, float64 -> float16 equivalent (5 exponent, 10 mantissa)
    operand = np.random.randn(2, 4, 4).astype(np.float64)
    input_dict = {"operand": operand, "exponent_bits": 5, "mantissa_bits": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, float32 -> (8 exponent, 0 mantissa)
    operand = np.array([1.5, 2.75, 3.125, -5.5], dtype=np.float32)
    input_dict = {"operand": operand, "exponent_bits": 8, "mantissa_bits": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reduce_precision_1"] = reduce_precision_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_precision_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_precision_1'.")


check_valid('jax.lax.reduce_precision', generated_inputs['jax.lax.reduce_precision_1'], lib="jax", suffix=1)
