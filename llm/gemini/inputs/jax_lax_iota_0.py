
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def iota_inputs():
    list_of_inputs = []

    # Input 1: int32, size 0
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.int32), "size": 0})
    )

    # Input 2: int32, size 1
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.int32), "size": 1})
    )

    # Input 3: int64, size 10
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.int64), "size": 10})
    )

    # Input 4: float32, size 5
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.float32), "size": 5})
    )

    # Input 5: float64, size 20
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.float64), "size": 20})
    )

    # Input 6: uint8, size 256
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.uint8), "size": 256})
    )

    # Input 7: uint32, size 100
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.uint32), "size": 100})
    )

    # Input 8: int8, size 128
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.int8), "size": 128})
    )

    # Input 9: int16, size 1024
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.int16), "size": 1024})
    )

    # Input 10: uint16, size 50
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.uint16), "size": 50})
    )

    # Input 11: bfloat16 or float16, size 64
    list_of_inputs.append(
        copy.deepcopy({"dtype": np.dtype(np.float16), "size": 64})
    )

    return list_of_inputs


generated_inputs["jax.lax.iota"] = iota_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.iota' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.iota'.")


check_valid('jax.lax.iota', generated_inputs['jax.lax.iota'], lib="jax", suffix=0)
