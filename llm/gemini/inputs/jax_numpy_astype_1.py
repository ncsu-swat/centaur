
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def astype_inputs():
    list_of_inputs = []

    # Input 1: 1D float array to int32, copy=False
    x = np.array([-1.5, 0.0, 1.5, 2.7], dtype=np.float32)
    dtype = np.int32
    input_dict = {"x": x, "dtype": dtype, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D integer array (with negatives) to float64, copy=True
    x = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    dtype = np.float64
    input_dict = {"x": x, "dtype": dtype, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean array to float32, copy=False
    x = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    dtype = np.float32
    input_dict = {"x": x, "dtype": dtype, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D array (scalar) to int32, copy=True
    x = np.array(5.5, dtype=np.float64)
    dtype = np.int32
    input_dict = {"x": x, "dtype": dtype, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D random float array to float16, copy=False
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    dtype = np.float16
    input_dict = {"x": x, "dtype": dtype, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex array to float32, copy=True
    x = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    dtype = np.float32
    input_dict = {"x": x, "dtype": dtype, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D large integer array to bool, copy=False
    x = np.array([[0, 100], [-50, 0]], dtype=np.int64)
    dtype = np.bool_
    input_dict = {"x": x, "dtype": dtype, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D unsigned int array to int16, copy=True
    x = np.array([[[0, 255]], [[128, 64]]], dtype=np.uint8)
    dtype = np.int16
    input_dict = {"x": x, "dtype": dtype, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array of zeros to int64, copy=False
    x = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
    dtype = np.int64
    input_dict = {"x": x, "dtype": dtype, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array to complex128, copy=True
    x = np.array([-1.23, 4.56, 7.89], dtype=np.float64)
    dtype = np.complex128
    input_dict = {"x": x, "dtype": dtype, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.astype_1"] = astype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.astype_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.astype_1'.")


check_valid('jax.numpy.astype', generated_inputs['jax.numpy.astype_1'], lib="jax", suffix=1)
