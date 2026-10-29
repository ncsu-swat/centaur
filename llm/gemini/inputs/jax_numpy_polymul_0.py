
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polymul_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays with trim_leading_zeros=False
    a1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a2 = np.array([4.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": False
    }))

    # Input 2: Leading zeros with trim_leading_zeros=True
    a1 = np.array([0.0, 2.0, 1.0, 0.0], dtype=np.float32)
    a2 = np.array([0.0, 5.0, 0.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": True
    }))

    # Input 3: Negative integer values
    a1 = np.array([-1, 3, -5], dtype=np.int32)
    a2 = np.array([2, -4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": False
    }))

    # Input 4: Float64 inputs with mixed positive/negative values
    a1 = np.array([1.5, -2.5, 0.5], dtype=np.float64)
    a2 = np.array([-3.0, 4.0, -1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": True
    }))

    # Input 5: Single element arrays (scalars as 1D tensors)
    a1 = np.array([5.0], dtype=np.float32)
    a2 = np.array([-3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": False
    }))

    # Input 6: Complex numbers (complex64)
    a1 = np.array([2.0 + 1.0j, 1.0 - 2.0j], dtype=np.complex64)
    a2 = np.array([3.0 + 0.0j, -1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": False
    }))

    # Input 7: Complex numbers with trim_leading_zeros=True
    a1 = np.array([0.0 + 0.0j, 1.0 + 2.0j, 3.0 + 0.0j], dtype=np.complex128)
    a2 = np.array([2.0 - 1.0j, 4.0 + 0.0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": True
    }))

    # Input 8: Highly asymmetric lengths
    a1 = np.array([1.0, 0.0, 2.0, -1.0, 3.0, -2.0, 1.0], dtype=np.float32)
    a2 = np.array([2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": False
    }))

    # Input 9: All zeros polynomial
    a1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    a2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": True
    }))

    # Input 10: Int64 inputs of different sizes
    a1 = np.array([10, -20, 30, -40], dtype=np.int64)
    a2 = np.array([1, -1, 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": False
    }))

    # Input 11: Longer 1D polynomials
    a1 = np.arange(10, dtype=np.float32)
    a2 = np.arange(-5, 5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "a1": a1,
        "a2": a2,
        "trim_leading_zeros": True
    }))

    return list_of_inputs

generated_inputs["jax.numpy.polymul"] = polymul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polymul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polymul'.")


check_valid('jax.numpy.polymul', generated_inputs['jax.numpy.polymul'], lib="jax", suffix=0)
