
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_precision_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array mimicking bfloat16 (8 exponent bits, 7 mantissa bits)
    operand = np.array([-1.5, 0.0, 2.75, 42.125], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "exponent_bits": 8,
        "mantissa_bits": 7,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array mimicking float16 (5 exponent bits, 10 mantissa bits)
    operand = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "exponent_bits": 5,
        "mantissa_bits": 10,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar (0D array) float32
    operand = np.array(3.1415927, dtype=np.float32)
    input_dict = {
        "operand": operand,
        "exponent_bits": 8,
        "mantissa_bits": 23,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array with lower precision
    operand = np.random.uniform(-10.0, 10.0, size=(2, 3, 2)).astype(np.float32)
    input_dict = {
        "operand": operand,
        "exponent_bits": 4,
        "mantissa_bits": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "operand": operand,
        "exponent_bits": 6,
        "mantissa_bits": 8,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with zero mantissa bits
    operand = np.linspace(-5.0, 5.0, num=10, dtype=np.float32)
    input_dict = {
        "operand": operand,
        "exponent_bits": 8,
        "mantissa_bits": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with minimal exponent bits
    operand = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "exponent_bits": 1,
        "mantissa_bits": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array with small exponent and mantissa
    operand = np.random.randn(1, 4, 1).astype(np.float32)
    input_dict = {
        "operand": operand,
        "exponent_bits": 3,
        "mantissa_bits": 4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with large values and extreme reduction
    operand = np.array([100.0, -200.0, 300.0, -400.0], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "exponent_bits": 7,
        "mantissa_bits": 14,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array
    operand = np.ones((1, 2, 1, 3, 2), dtype=np.float32) * 1.5
    input_dict = {
        "operand": operand,
        "exponent_bits": 4,
        "mantissa_bits": 5,
    }
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
