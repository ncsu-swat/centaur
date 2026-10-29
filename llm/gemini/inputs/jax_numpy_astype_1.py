
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def astype_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 to float32, copy=False
    x = np.array([0, -1, 2, -3, 4], dtype=np.int32)
    dtype = np.dtype(np.float32)
    copy_val = False
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 2: 2D float32 to int32 with negative values, copy=True
    x = np.array([[-1.5, 2.7], [3.2, -4.9]], dtype=np.float32)
    dtype = np.dtype(np.int32)
    copy_val = True
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 3: 0D scalar float64 to bool, copy=False
    x = np.array(-0.0, dtype=np.float64)
    dtype = np.dtype(bool)
    copy_val = False
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 4: 3D int16 to float64, copy=True
    x = np.arange(-12, 12, dtype=np.int16).reshape(2, 3, 4)
    dtype = np.dtype(np.float64)
    copy_val = True
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 5: 1D bool to int8, copy=False
    x = np.array([True, False, True, True], dtype=bool)
    dtype = np.dtype(np.int8)
    copy_val = False
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 6: 4D float32 to float16, copy=False
    x = np.random.randn(2, 2, 3, 2).astype(np.float32)
    dtype = np.dtype(np.float16)
    copy_val = False
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 7: 2D uint8 to float32, copy=True
    x = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    dtype = np.dtype(np.float32)
    copy_val = True
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 8: 1D complex64 to complex128, copy=False
    x = np.array([1.0 - 2.0j, -3.5 + 0.5j], dtype=np.complex64)
    dtype = np.dtype(np.complex128)
    copy_val = False
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 9: 2D int64 to uint32, copy=True
    x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    dtype = np.dtype(np.uint32)
    copy_val = True
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 10: 1D float64 to float32 (same family cast), copy=True
    x = np.array([-1e-5, 0.0, 1e5], dtype=np.float64)
    dtype = np.dtype(np.float32)
    copy_val = True
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

    # Input 11: 3D int32 to int64, copy=False
    x = np.ones((2, 2, 2), dtype=np.int32) * -5
    dtype = np.dtype(np.int64)
    copy_val = False
    list_of_inputs.append({"x": copy.deepcopy(x), "dtype": dtype, "copy": copy_val})

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
