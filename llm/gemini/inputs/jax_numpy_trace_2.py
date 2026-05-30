
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trace_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 trace, offset=0
    a = np.random.randn(4, 4).astype(np.float32)
    offset = np.array(0, dtype=np.int32)
    axis1 = 0
    axis2 = 1
    dtype = np.float32
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 trace with positive offset
    a = np.random.randn(5, 5).astype(np.float32)
    offset = np.array(1, dtype=np.int32)
    axis1 = 0
    axis2 = 1
    dtype = np.float32
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 trace with negative offset
    a = np.random.randn(6, 6).astype(np.float64)
    offset = np.array(-2, dtype=np.int32)
    axis1 = 0
    axis2 = 1
    dtype = np.float64
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 trace on axes 1 and 2
    a = np.random.randn(3, 4, 4).astype(np.float32)
    offset = np.array(0, dtype=np.int32)
    axis1 = 1
    axis2 = 2
    dtype = np.float32
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 trace on axes 0 and 2
    a = np.random.randn(4, 3, 4).astype(np.float32)
    offset = np.array(1, dtype=np.int32)
    axis1 = 0
    axis2 = 2
    dtype = np.float32
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer array with cast to int64
    a = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    offset = np.array(0, dtype=np.int32)
    axis1 = 0
    axis2 = 1
    dtype = np.int64
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Upcasting float32 input to float64 output
    a = np.random.randn(4, 4).astype(np.float32)
    offset = np.array(-1, dtype=np.int32)
    axis1 = 0
    axis2 = 1
    dtype = np.float64
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rectangular matrix trace
    a = np.random.randn(3, 6).astype(np.float32)
    offset = np.array(2, dtype=np.int32)
    axis1 = 0
    axis2 = 1
    dtype = np.float32
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensor trace with offset
    a = np.random.randn(2, 2, 5, 5).astype(np.float32)
    offset = np.array(-3, dtype=np.int32)
    axis1 = 2
    axis2 = 3
    dtype = np.float32
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 2D array, float16 to float32 trace
    a = np.random.randn(10, 10).astype(np.float16)
    offset = np.array(0, dtype=np.int32)
    axis1 = 0
    axis2 = 1
    dtype = np.float32
    input_dict = {"a": a, "offset": offset, "axis1": axis1, "axis2": axis2, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.trace_2"] = trace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trace_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trace_2'.")


check_valid('jax.numpy.trace', generated_inputs['jax.numpy.trace_2'], lib="jax", suffix=2)
