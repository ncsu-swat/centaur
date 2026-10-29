
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def can_cast_inputs():
    list_of_inputs = []

    # Input 1: int32 to int64, safe casting
    input_dict = {
        "from_": "int32",
        "to": "int64",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 to float64, safe casting
    input_dict = {
        "from_": "float32",
        "to": "float64",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 to float32, same_kind casting
    input_dict = {
        "from_": "float64",
        "to": "float32",
        "casting": "same_kind"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 to int32, unsafe casting
    input_dict = {
        "from_": "int64",
        "to": "int32",
        "casting": "unsafe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32 to int32, no casting
    input_dict = {
        "from_": "int32",
        "to": "int32",
        "casting": "no"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 to float32, equiv casting
    input_dict = {
        "from_": "float32",
        "to": "float32",
        "casting": "equiv"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bool to int32, safe casting
    input_dict = {
        "from_": "bool",
        "to": "int32",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32 to complex64, safe casting
    input_dict = {
        "from_": "int32",
        "to": "complex64",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8 to int16, safe casting
    input_dict = {
        "from_": "uint8",
        "to": "int16",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 to int64, unsafe casting
    input_dict = {
        "from_": "float64",
        "to": "int64",
        "casting": "unsafe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: complex128 to complex64, same_kind casting
    input_dict = {
        "from_": "complex128",
        "to": "complex64",
        "casting": "same_kind"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: int16 to float32, safe casting
    input_dict = {
        "from_": "int16",
        "to": "float32",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.can_cast_4"] = can_cast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.can_cast_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.can_cast_4'.")


check_valid('jax.numpy.can_cast', generated_inputs['jax.numpy.can_cast_4'], lib="jax", suffix=4)
