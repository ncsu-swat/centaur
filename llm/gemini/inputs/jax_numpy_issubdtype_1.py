
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def issubdtype_inputs():
    list_of_inputs = []

    # Input 1: int32 is subdtype of int32
    input_dict = {"arg1": np.dtype("int32"), "arg2": np.dtype("int32")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 vs float64
    input_dict = {"arg1": np.dtype("float32"), "arg2": np.dtype("float64")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bool is subdtype of bool
    input_dict = {"arg1": np.dtype("bool"), "arg2": np.dtype("bool")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64 is subdtype of complex64
    input_dict = {"arg1": np.dtype("complex64"), "arg2": np.dtype("complex64")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8 is subdtype of uint8
    input_dict = {"arg1": np.dtype("uint8"), "arg2": np.dtype("uint8")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int16 vs int32
    input_dict = {"arg1": np.dtype("int16"), "arg2": np.dtype("int32")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 is subdtype of float64
    input_dict = {"arg1": np.dtype("float64"), "arg2": np.dtype("float64")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128 vs float64
    input_dict = {"arg1": np.dtype("complex128"), "arg2": np.dtype("float64")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint64 is subdtype of uint64
    input_dict = {"arg1": np.dtype("uint64"), "arg2": np.dtype("uint64")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int8 is subdtype of int8
    input_dict = {"arg1": np.dtype("int8"), "arg2": np.dtype("int8")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float16 is subdtype of float16
    input_dict = {"arg1": np.dtype("float16"), "arg2": np.dtype("float16")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: int64 is subdtype of int64
    input_dict = {"arg1": np.dtype("int64"), "arg2": np.dtype("int64")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.issubdtype_1"] = issubdtype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.issubdtype_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.issubdtype_1'.")


check_valid('jax.numpy.issubdtype', generated_inputs['jax.numpy.issubdtype_1'], lib="jax", suffix=1)
