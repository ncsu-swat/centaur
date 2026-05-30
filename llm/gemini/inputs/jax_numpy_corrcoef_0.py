
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def corrcoef_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, rowvar=True
    x = np.random.randn(10).astype(np.float32)
    y = np.random.randn(10).astype(np.float32)
    input_dict = {"x": x, "y": y, "rowvar": True, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float64, rowvar=False
    x = np.random.randn(15).astype(np.float64)
    y = np.random.randn(15).astype(np.float64)
    input_dict = {"x": x, "y": y, "rowvar": False, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float32, rowvar=True, matching columns (N=5)
    x = np.random.randn(3, 5).astype(np.float32)
    y = np.random.randn(2, 5).astype(np.float32)
    input_dict = {"x": x, "y": y, "rowvar": True, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, float32, rowvar=False, matching rows (N=6)
    x = np.random.randn(6, 2).astype(np.float32)
    y = np.random.randn(6, 4).astype(np.float32)
    input_dict = {"x": x, "y": y, "rowvar": False, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D arrays, float64, negative values, rowvar=True
    x = np.array([-1.0, 0.0, 1.0, 2.0, -3.0], dtype=np.float64)
    y = np.array([5.0, -2.0, 1.0, 0.0, 4.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "rowvar": True, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D arrays, complex64, rowvar=True
    x = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    y = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    input_dict = {"x": x, "y": y, "rowvar": True, "dtype": np.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D arrays, complex128, rowvar=False, matching rows (N=4)
    x = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(np.complex128)
    y = (np.random.randn(4, 2) + 1j * np.random.randn(4, 2)).astype(np.complex128)
    input_dict = {"x": x, "y": y, "rowvar": False, "dtype": np.complex128}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D arrays, large N, float32, rowvar=True
    x = np.random.randn(1000).astype(np.float32)
    y = np.random.randn(1000).astype(np.float32)
    input_dict = {"x": x, "y": y, "rowvar": True, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D arrays, float64, rowvar=True, single row for x and y
    x = np.random.randn(1, 50).astype(np.float64)
    y = np.random.randn(1, 50).astype(np.float64)
    input_dict = {"x": x, "y": y, "rowvar": True, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D arrays, float32, rowvar=False, single col for x and y
    x = np.random.randn(50, 1).astype(np.float32)
    y = np.random.randn(50, 1).astype(np.float32)
    input_dict = {"x": x, "y": y, "rowvar": False, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.corrcoef"] = corrcoef_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.corrcoef' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.corrcoef'.")


check_valid('jax.numpy.corrcoef', generated_inputs['jax.numpy.corrcoef'], lib="jax", suffix=0)
