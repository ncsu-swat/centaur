
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isdtype_inputs():
    list_of_inputs = []

    # Input 1: bool with 'bool'
    input_dict = {"dtype": np.dtype(np.bool_), "kind": "bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 with 'signed integer'
    input_dict = {"dtype": np.dtype(np.int32), "kind": "signed integer"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64 with 'integral'
    input_dict = {"dtype": np.dtype(np.int64), "kind": "integral"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8 with 'unsigned integer'
    input_dict = {"dtype": np.dtype(np.uint8), "kind": "unsigned integer"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint32 with 'integral'
    input_dict = {"dtype": np.dtype(np.uint32), "kind": "integral"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 with 'real floating'
    input_dict = {"dtype": np.dtype(np.float32), "kind": "real floating"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 with 'numeric'
    input_dict = {"dtype": np.dtype(np.float64), "kind": "numeric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64 with 'complex floating'
    input_dict = {"dtype": np.dtype(np.complex64), "kind": "complex floating"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128 with 'numeric'
    input_dict = {"dtype": np.dtype(np.complex128), "kind": "numeric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16 tested against 'real floating' (evaluates to False)
    input_dict = {"dtype": np.dtype(np.int16), "kind": "real floating"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float16 tested against 'signed integer' (evaluates to False)
    input_dict = {"dtype": np.dtype(np.float16), "kind": "signed integer"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: uint64 tested against 'numeric'
    input_dict = {"dtype": np.dtype(np.uint64), "kind": "numeric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isdtype_1"] = isdtype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isdtype_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isdtype_1'.")


check_valid('jax.numpy.isdtype', generated_inputs['jax.numpy.isdtype_1'], lib="jax", suffix=1)
