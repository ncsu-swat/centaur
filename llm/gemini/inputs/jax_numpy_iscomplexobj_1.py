
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D complex128 array
    x = np.array([[-1.5 - 2.5j, 3.0j], [4.0 + 0.0j, -5.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D float32 array (real)
    x = np.array([1.0, -2.5, 3.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D float64 array
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 0D scalar complex array
    x = np.array(2.0 - 3.0j, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 0D scalar integer array
    x = np.array(-42, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D int64 array
    x = np.arange(8, dtype=np.int64).reshape((2, 2, 2)) - 4
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D boolean array
    x = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty 1D complex array
    x = np.array([], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 2D float array
    x = np.empty((0, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 4D complex array
    x = np.ones((2, 1, 3, 2), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: 1D complex array with zero imaginary components
    x = np.array([1.0 + 0.0j, -2.0 + 0.0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.iscomplexobj_1"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_1'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_1'], lib="jax", suffix=1)
