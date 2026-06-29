
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trace_inputs():
    list_of_inputs = []

    # Input 1: 2D float32, offset 0, dtype float32
    x = np.random.randn(4, 4).astype(np.float32)
    input_dict = {"x": x, "offset": 0, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32, offset 1, dtype float64
    x = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    input_dict = {"x": x, "offset": 1, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64, offset -1, dtype float32
    x = np.random.randn(3, 5).astype(np.float64)
    input_dict = {"x": x, "offset": -1, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 (batched), offset 0, dtype float32
    x = np.random.randn(2, 4, 4).astype(np.float32)
    input_dict = {"x": x, "offset": 0, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64, offset 2, dtype float64
    x = np.random.randn(3, 6, 6).astype(np.float64)
    input_dict = {"x": x, "offset": 2, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32, offset -2, dtype float32
    x = np.random.randn(2, 3, 5, 5).astype(np.float32)
    input_dict = {"x": x, "offset": -2, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex64, offset 0, dtype complex128
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {"x": x, "offset": 0, "dtype": np.complex128}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32, offset -3, dtype float64
    x = np.random.randn(6, 4).astype(np.float32)
    input_dict = {"x": x, "offset": -3, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int32, offset 1, dtype int64
    x = np.random.randint(-5, 5, size=(2, 3, 3)).astype(np.int32)
    input_dict = {"x": x, "offset": 1, "dtype": np.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float64, offset 3, dtype float64
    x = np.random.randn(2, 2, 7, 7).astype(np.float64)
    input_dict = {"x": x, "offset": 3, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.trace"] = trace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.trace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.trace'.")


check_valid('jax.numpy.linalg.trace', generated_inputs['jax.numpy.linalg.trace'], lib="jax", suffix=0)
