
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_linalg_outer_inputs():
    list_of_inputs = []

    # Input 1: Float32, same size
    x1 = np.random.randn(5).astype(np.float32)
    x2 = np.random.randn(5).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, different size
    x1 = np.random.randn(10).astype(np.float64)
    x2 = np.random.randn(8).astype(np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int32, with negative values
    x1 = np.random.randint(-10, 10, size=6).astype(np.int32)
    x2 = np.random.randint(-10, 10, size=6).astype(np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int64, different size
    x1 = np.random.randint(-100, 100, size=4).astype(np.int64)
    x2 = np.random.randint(-100, 100, size=12).astype(np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float16
    x1 = np.random.randn(7).astype(np.float16)
    x2 = np.random.randn(7).astype(np.float16)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64
    x1 = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    x2 = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float32, single element arrays
    x1 = np.array([2.5], dtype=np.float32)
    x2 = np.array([-1.5], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: UInt8
    x1 = np.random.randint(0, 10, size=5).astype(np.uint8)
    x2 = np.random.randint(0, 10, size=5).astype(np.uint8)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large arrays (Float32)
    x1 = np.random.randn(100).astype(np.float32)
    x2 = np.random.randn(200).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float32, with zeros
    x1 = np.zeros(8, dtype=np.float32)
    x2 = np.random.randn(8).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Int16, mixed signs
    x1 = np.random.randint(-50, 50, size=9).astype(np.int16)
    x2 = np.random.randint(-50, 50, size=9).astype(np.int16)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.outer"] = jax_numpy_linalg_outer_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.outer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.outer'.")


check_valid('jax.numpy.linalg.outer', generated_inputs['jax.numpy.linalg.outer'], lib="jax", suffix=0)
