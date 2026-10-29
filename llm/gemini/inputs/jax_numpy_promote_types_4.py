
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def promote_types_inputs():
    list_of_inputs = []

    test_cases = [
        ("int32", np.dtype(np.float32)),
        ("float16", np.dtype(np.float32)),
        ("bool", np.dtype(np.int8)),
        ("uint8", np.dtype(np.int32)),
        ("int64", np.dtype(np.float64)),
        ("float32", np.dtype(np.complex64)),
        ("complex64", np.dtype(np.float64)),
        ("uint16", np.dtype(np.uint32)),
        ("int16", np.dtype(np.float16)),
        ("bool", np.dtype(np.bool_)),
        ("int8", np.dtype(np.uint8)),
        ("float64", np.dtype(np.complex128)),
    ]

    for a, b in test_cases:
        input_dict = {"a": a, "b": b}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.promote_types_4"] = promote_types_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.promote_types_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.promote_types_4'.")


check_valid('jax.numpy.promote_types', generated_inputs['jax.numpy.promote_types_4'], lib="jax", suffix=4)
