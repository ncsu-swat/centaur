
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def empty_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 prototype, 2D shape, int32 dtype
    prototype = np.random.randn(5).astype(np.float32)
    dtype = np.dtype(np.int32)
    shape = (2, 3)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 prototype, 3D shape, float64 dtype
    prototype = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    dtype = np.dtype(np.float64)
    shape = (2, 2, 2)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D bool prototype, 1D shape, complex64 dtype
    prototype = np.random.choice([True, False], size=(2, 2, 2))
    dtype = np.dtype(np.complex64)
    shape = (10,)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float64 prototype, 2D shape, bool dtype
    prototype = np.random.randn(2, 2, 2, 2).astype(np.float64)
    dtype = np.dtype(bool)
    shape = (4, 4)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D float32 prototype, 3D shape, int16 dtype
    prototype = np.array(3.14, dtype=np.float32)
    dtype = np.dtype(np.int16)
    shape = (1, 5, 5)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex128 prototype, 1D shape, float32 dtype
    prototype = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex128)
    dtype = np.dtype(np.float32)
    shape = (8,)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D uint8 prototype, 4D shape, uint32 dtype
    prototype = np.random.randint(0, 255, size=(4, 4)).astype(np.uint8)
    dtype = np.dtype(np.uint32)
    shape = (2, 2, 2, 2)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int16 prototype, 2D shape, int64 dtype
    prototype = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int16)
    dtype = np.dtype(np.int64)
    shape = (5, 2)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float16 prototype, 3D shape, complex128 dtype
    prototype = np.random.randn(10).astype(np.float16)
    dtype = np.dtype(np.complex128)
    shape = (3, 3, 3)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 prototype, 1D shape, uint8 dtype
    prototype = np.random.randn(10, 10).astype(np.float64)
    dtype = np.dtype(np.uint8)
    shape = (100,)
    input_dict = {"prototype": prototype, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.empty_like_1"] = empty_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_like_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_like_1'.")


check_valid('jax.numpy.empty_like', generated_inputs['jax.numpy.empty_like_1'], lib="jax", suffix=1)
