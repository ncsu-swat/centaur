
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def astype_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array to float32, copy=False
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    dtype = "float32"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 2: 2D float32 array to int32, copy=True
    x = np.array([[-3.7, 2.5], [0.0, -1.2]], dtype=np.float32)
    dtype = "int32"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 3: 0D scalar array to bool, copy=False
    x = np.array(5.0, dtype=np.float32)
    dtype = "bool"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 4: 3D int32 array to float32, copy=True
    x = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    dtype = "float32"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 5: 1D bool array to int8, copy=False
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    dtype = "int8"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 6: 2D float32 array to complex64, copy=True
    x = np.array([[-1.0, 2.0], [3.5, -4.5]], dtype=np.float32)
    dtype = "complex64"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 7: 4D float32 array to uint8, copy=False
    x = np.ones((2, 2, 2, 2), dtype=np.float32) * 128.0
    dtype = "uint8"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 8: 1D float32 array to float16, copy=True
    x = np.array([-1e-4, 0.0, 1e-4, 3.14], dtype=np.float32)
    dtype = "float16"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 9: 2D int32 array to int16, copy=False
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    dtype = "int16"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 10: 2D empty float32 array to int32, copy=True
    x = np.empty((0, 5), dtype=np.float32)
    dtype = "int32"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    # Input 11: 3D float32 array to uint32, copy=False
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    dtype = "uint32"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "dtype": dtype, "copy": copy_flag})
    )

    return list_of_inputs


generated_inputs["jax.numpy.astype_2"] = astype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.astype_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.astype_2'.")


check_valid('jax.numpy.astype', generated_inputs['jax.numpy.astype_2'], lib="jax", suffix=2)
