
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hstack_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, same size
    tup1 = (np.random.randn(5).astype(np.float32), np.random.randn(5).astype(np.float32))
    input_dict = {"tup": tup1, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float32, same size (larger)
    tup2 = (np.random.randn(10).astype(np.float32), np.random.randn(10).astype(np.float32))
    input_dict = {"tup": tup2, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays with negative values, float64, same size
    tup3 = (np.array([-1.5, -2.0, 3.5], dtype=np.float64), np.array([4.0, -5.5, -1.0], dtype=np.float64))
    input_dict = {"tup": tup3, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, float64, same size
    tup4 = (np.random.randn(3, 4).astype(np.float64), np.random.randn(3, 4).astype(np.float64))
    input_dict = {"tup": tup4, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays, int32, same size
    tup5 = (np.random.randint(-10, 10, size=(2, 3)).astype(np.int32), np.random.randint(-10, 10, size=(2, 3)).astype(np.int32))
    input_dict = {"tup": tup5, "dtype": np.dtype('int32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays, float32, same size
    tup6 = (np.random.randn(2, 3, 4).astype(np.float32), np.random.randn(2, 3, 4).astype(np.float32))
    input_dict = {"tup": tup6, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D arrays with int16 dtype, same size, output dtype specified as int32
    tup7 = (np.array([1, 2, 3, 4], dtype=np.int16), np.array([5, 6, 7, 8], dtype=np.int16))
    input_dict = {"tup": tup7, "dtype": np.dtype('int32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D arrays of length 1, float32, output dtype float64
    tup8 = (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))
    input_dict = {"tup": tup8, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D arrays, float32, same size
    tup9 = (np.random.randn(2, 2, 2, 2).astype(np.float32), np.random.randn(2, 2, 2, 2).astype(np.float32))
    input_dict = {"tup": tup9, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D arrays, boolean, same size, output dtype int32
    tup10 = (np.array([[True, False], [False, True]]), np.array([[True, True], [False, False]]))
    input_dict = {"tup": tup10, "dtype": np.dtype('int32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.hstack_2"] = hstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hstack_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hstack_2'.")


check_valid('jax.numpy.hstack', generated_inputs['jax.numpy.hstack_2'], lib="jax", suffix=2)
