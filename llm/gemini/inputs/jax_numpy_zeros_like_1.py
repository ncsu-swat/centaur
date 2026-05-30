
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, shape (5,)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    dtype = np.float32
    shape = (5,)
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, shape (3, 4)
    a = np.random.randint(-10, 10, size=(2, 2)).astype(np.int32)
    dtype = np.int32
    shape = (3, 4)
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D bool array, shape (2, 2, 2)
    a = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    dtype = np.bool_
    shape = (2, 2, 2)
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float64 array, shape (1, 2, 3, 4)
    a = np.random.randn(2, 2, 2, 2).astype(np.float64)
    dtype = np.float64
    shape = (1, 2, 3, 4)
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D int64 array, shape ()
    a = np.array(42, dtype=np.int64)
    dtype = np.int64
    shape = ()
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex64 array, shape (10,)
    a = np.array([1+2j, 3+4j], dtype=np.complex64)
    dtype = np.complex64
    shape = (10,)
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float16 array, shape (8, 8)
    a = np.ones((4, 4), dtype=np.float16)
    dtype = np.float16
    shape = (8, 8)
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int8 array, shape (4, 3, 2)
    a = np.zeros((2, 2, 2), dtype=np.int8)
    dtype = np.int8
    shape = (4, 3, 2)
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 array, shape (100,)
    a = np.arange(10, dtype=np.float32)
    dtype = np.float64
    shape = (100,)
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int16 array, shape (5, 5)
    a = np.array([[1, 2], [3, 4]], dtype=np.int16)
    dtype = np.bool_
    shape = (5, 5)
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.zeros_like_1"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_1'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_1'], lib="jax", suffix=1)
