
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def empty_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 -> 1D float32
    prototype = np.random.randn(5).astype(np.float32)
    dtype = np.dtype(np.float32)
    shape = [5]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 -> 2D int32
    prototype = np.random.randn(2, 3).astype(np.float32)
    dtype = np.dtype(np.int32)
    shape = [3, 4]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 -> 3D complex64
    prototype = np.random.randn(2, 2, 2).astype(np.float64)
    dtype = np.dtype(np.complex64)
    shape = [2, 3, 4]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D int32 -> 4D bool
    prototype = np.random.randint(0, 10, size=(1, 2, 3, 4)).astype(np.int32)
    dtype = np.dtype(np.bool_)
    shape = [2, 2, 2, 2]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D bool -> 1D float64
    prototype = np.array([True, False, True], dtype=np.bool_)
    dtype = np.dtype(np.float64)
    shape = [10]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex64 -> 2D float32
    prototype = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    dtype = np.dtype(np.float32)
    shape = [5, 5]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int64 -> 1D int8
    prototype = np.random.randint(-100, 100, size=(10,)).astype(np.int64)
    dtype = np.dtype(np.int8)
    shape = [20]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D uint8 -> 3D uint32
    prototype = np.random.randint(0, 255, size=(2, 2, 2)).astype(np.uint8)
    dtype = np.dtype(np.uint32)
    shape = [4, 4, 4]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int16 -> 2D complex128
    prototype = np.random.randint(-10, 10, size=(4, 4)).astype(np.int16)
    dtype = np.dtype(np.complex128)
    shape = [1, 10]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 -> 1D int64
    prototype = np.random.randn(8).astype(np.float64)
    dtype = np.dtype(np.int64)
    shape = [100]
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.empty_like_3"] = empty_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_like_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_like_3'.")


check_valid('jax.numpy.empty_like', generated_inputs['jax.numpy.empty_like_3'], lib="jax", suffix=3)
