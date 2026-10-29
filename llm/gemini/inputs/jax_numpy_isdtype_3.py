
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np


def isdtype_inputs():
    list_of_inputs = []

    # Input 1: float32
    list_of_inputs.append({"dtype": np.dtype(np.float32), "kind": ()})

    # Input 2: float64
    list_of_inputs.append({"dtype": np.dtype(np.float64), "kind": ()})

    # Input 3: int32
    list_of_inputs.append({"dtype": np.dtype(np.int32), "kind": ()})

    # Input 4: int64
    list_of_inputs.append({"dtype": np.dtype(np.int64), "kind": ()})

    # Input 5: bool
    list_of_inputs.append({"dtype": np.dtype(np.bool_), "kind": ()})

    # Input 6: uint8
    list_of_inputs.append({"dtype": np.dtype(np.uint8), "kind": ()})

    # Input 7: uint32
    list_of_inputs.append({"dtype": np.dtype(np.uint32), "kind": ()})

    # Input 8: complex64
    list_of_inputs.append({"dtype": np.dtype(np.complex64), "kind": ()})

    # Input 9: complex128
    list_of_inputs.append({"dtype": np.dtype(np.complex128), "kind": ()})

    # Input 10: int16
    list_of_inputs.append({"dtype": np.dtype(np.int16), "kind": ()})

    # Input 11: uint16
    list_of_inputs.append({"dtype": np.dtype(np.uint16), "kind": ()})

    return list_of_inputs


generated_inputs["jax.numpy.isdtype_3"] = isdtype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isdtype_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isdtype_3'.")


check_valid('jax.numpy.isdtype', generated_inputs['jax.numpy.isdtype_3'], lib="jax", suffix=3)
