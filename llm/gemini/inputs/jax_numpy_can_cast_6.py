
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def can_cast_inputs():
    list_of_inputs = []

    # Input 1, float32 array to float64, safe casting
    from_ = np.array([[1.0, -2.0], [3.5, -4.5]], dtype=np.float32)
    input_dict = {
        "from_": from_,
        "to": "float64",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, int32 array to int64, safe casting
    from_ = np.array([-10, 0, 10], dtype=np.int32)
    input_dict = {
        "from_": from_,
        "to": "int64",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, float64 array to float32, unsafe casting
    from_ = np.array([1.5], dtype=np.float64)
    input_dict = {
        "from_": from_,
        "to": "float32",
        "casting": "unsafe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, 0-D int16 array to int8, same_kind casting
    from_ = np.array(-5, dtype=np.int16)
    input_dict = {
        "from_": from_,
        "to": "int8",
        "casting": "same_kind"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, complex64 array to complex128, safe casting
    from_ = np.array([[[1+2j, -3j]]], dtype=np.complex64)
    input_dict = {
        "from_": from_,
        "to": "complex128",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, uint8 array to int32, safe casting
    from_ = np.array([[0, 255]], dtype=np.uint8)
    input_dict = {
        "from_": from_,
        "to": "int32",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, bool array to int8, safe casting
    from_ = np.array([True, False, True], dtype=np.bool_)
    input_dict = {
        "from_": from_,
        "to": "int8",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, float32 array to int32, unsafe casting
    from_ = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "from_": from_,
        "to": "int32",
        "casting": "unsafe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, int64 array to float64, safe casting
    from_ = np.array([[-123456789, 987654321]], dtype=np.int64)
    input_dict = {
        "from_": from_,
        "to": "float64",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, int32 array to int32, no casting
    from_ = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "from_": from_,
        "to": "int32",
        "casting": "no"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.can_cast_6"] = can_cast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.can_cast_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.can_cast_6'.")


check_valid('jax.numpy.can_cast', generated_inputs['jax.numpy.can_cast_6'], lib="jax", suffix=6)
