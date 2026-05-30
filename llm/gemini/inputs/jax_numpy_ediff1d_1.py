
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ediff1d_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 arrays
    ary = np.array([1, 2, 4, 7, 11], dtype=np.int32)
    to_end = np.array([16, 22], dtype=np.int32)
    to_begin = np.array([0], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 2: 1D float32 arrays with negative values
    ary = np.array([-1.5, 2.5, -3.5, 4.5], dtype=np.float32)
    to_end = np.array([-10.0], dtype=np.float32)
    to_begin = np.array([10.0], dtype=np.float32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 3: 2D int64 array for ary
    ary = np.array([[1, 2], [3, 4]], dtype=np.int64)
    to_end = np.array([5, 6], dtype=np.int64)
    to_begin = np.array([0], dtype=np.int64)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 4: 3D float64 array for ary
    ary = np.arange(8, dtype=np.float64).reshape((2, 2, 2))
    to_end = np.array([8.0, 9.0], dtype=np.float64)
    to_begin = np.array([-1.0], dtype=np.float64)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 5: Scalar tensors (0D arrays) for to_end and to_begin
    ary = np.array([10, 20, 30], dtype=np.int32)
    to_end = np.array(40, dtype=np.int32)
    to_begin = np.array(0, dtype=np.int32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 6: Empty arrays for to_end and to_begin
    ary = np.array([1.0, 3.0, 6.0], dtype=np.float32)
    to_end = np.array([], dtype=np.float32)
    to_begin = np.array([], dtype=np.float32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 7: Empty array for ary
    ary = np.array([], dtype=np.int32)
    to_end = np.array([1, 2], dtype=np.int32)
    to_begin = np.array([-2, -1], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 8: High-dimensional array, float32
    ary = np.random.randn(2, 3, 4).astype(np.float32)
    to_end = np.array([0.5, 0.6], dtype=np.float32)
    to_begin = np.array([-0.5], dtype=np.float32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 9: Large 1D array, int64 dtype
    ary = np.arange(100, dtype=np.int64)
    to_end = np.array([100], dtype=np.int64)
    to_begin = np.array([-1], dtype=np.int64)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 10: float64 type
    ary = np.array([10.5, 20.5, 40.5], dtype=np.float64)
    to_end = np.array([80.5], dtype=np.float64)
    to_begin = np.array([5.5], dtype=np.float64)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    return list_of_inputs

generated_inputs["jax.numpy.ediff1d_1"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_1'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_1'], lib="jax", suffix=1)
