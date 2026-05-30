
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def empty_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, dtype float32, shape 5
    prototype = np.random.randn(10).astype(np.float32)
    dtype = np.dtype('float32')
    shape = 5
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, dtype int32, shape 3
    prototype = np.random.randint(-10, 10, size=(4, 4)).astype(np.int32)
    dtype = np.dtype('int32')
    shape = 3
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D bool array, dtype bool, shape 10
    prototype = np.random.choice([True, False], size=(2, 2, 2))
    dtype = np.dtype('bool')
    shape = 10
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array, dtype float64, shape 8
    prototype = np.random.randn(3).astype(np.float64)
    dtype = np.dtype('float64')
    shape = 8
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D int16 array, dtype int16, shape 1
    prototype = np.random.randint(-100, 100, size=(2, 2, 2, 2)).astype(np.int16)
    dtype = np.dtype('int16')
    shape = 1
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex64 array, dtype complex64, shape 2
    prototype = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    dtype = np.dtype('complex64')
    shape = 2
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D uint8 array, dtype uint8, shape 6
    prototype = np.random.randint(0, 255, size=(3, 3, 3)).astype(np.uint8)
    dtype = np.dtype('uint8')
    shape = 6
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int64 array, dtype int64, shape 12
    prototype = np.random.randint(-1000, 1000, size=(5,)).astype(np.int64)
    dtype = np.dtype('int64')
    shape = 12
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float16 array, dtype float16, shape 4
    prototype = np.random.randn(2, 5).astype(np.float16)
    dtype = np.dtype('float16')
    shape = 4
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D bool array, dtype bool, shape 7
    prototype = np.random.choice([True, False], size=(1, 2, 1, 2, 1))
    dtype = np.dtype('bool')
    shape = 7
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.empty_like_2"] = empty_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_like_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_like_2'.")


check_valid('jax.numpy.empty_like', generated_inputs['jax.numpy.empty_like_2'], lib="jax", suffix=2)
