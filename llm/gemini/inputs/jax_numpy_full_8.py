
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_inputs():
    list_of_inputs = []

    # Input 1: 1D shape, scalar tensor, float32 dtype
    shape = (5,)
    fill_value = np.array(3.14, dtype=np.float32)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 2: 2D shape, scalar tensor with negative int, int32 dtype
    shape = (2, 3)
    fill_value = np.array(-7, dtype=np.int32)
    dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 3: 3D shape, 1D broadcastable array fill_value, float64 dtype
    shape = (2, 3, 4)
    fill_value = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    dtype = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 4: 0D shape (scalar array), scalar tensor, int64 dtype
    shape = ()
    fill_value = np.array(42, dtype=np.int64)
    dtype = np.dtype(np.int64)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 5: 2D shape, bool fill_value, bool dtype
    shape = (4, 4)
    fill_value = np.array(True, dtype=np.bool_)
    dtype = np.dtype(np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 6: 4D shape, complex fill_value, complex64 dtype
    shape = (2, 1, 3, 2)
    fill_value = np.array(1.0 - 2.0j, dtype=np.complex64)
    dtype = np.dtype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 7: 2D shape, 2D broadcastable tensor (1, 3), float32 dtype
    shape = (4, 3)
    fill_value = np.array([[-1.5, 0.0, 1.5]], dtype=np.float32)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 8: 1D shape with 0 elements (empty array), float32 fill_value, float32 dtype
    shape = (0,)
    fill_value = np.array(0.0, dtype=np.float32)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 9: 2D shape with uint8 dtype, scalar tensor fill_value
    shape = (3, 5)
    fill_value = np.array(255, dtype=np.uint8)
    dtype = np.dtype(np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 10: 3D shape, 2D broadcastable tensor (3, 1), int16 dtype
    shape = (2, 3, 4)
    fill_value = np.array([[10], [-20], [30]], dtype=np.int16)
    dtype = np.dtype(np.int16)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    # Input 11: 1D shape, complex128 fill_value, complex128 dtype
    shape = (6,)
    fill_value = np.array(-0.5 + 4.2j, dtype=np.complex128)
    dtype = np.dtype(np.complex128)
    list_of_inputs.append(
        copy.deepcopy({"shape": shape, "fill_value": fill_value, "dtype": dtype})
    )

    return list_of_inputs


generated_inputs["jax.numpy.full_8"] = full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_8'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_8'], lib="jax", suffix=8)
