
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def can_cast_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array to int64 with safe casting
    input_dict = {
        "from_": np.array([1, 2, 3], dtype=np.int32),
        "to": "int64",
        "casting": "safe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array to float64 with safe casting
    input_dict = {
        "from_": np.random.randn(3, 4).astype(np.float32),
        "to": "float64",
        "casting": "safe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D float64 array to float32 with same_kind casting
    input_dict = {
        "from_": np.array(3.14, dtype=np.float64),
        "to": "float32",
        "casting": "same_kind",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int64 array to int32 with unsafe casting
    input_dict = {
        "from_": np.ones((2, 3, 2), dtype=np.int64),
        "to": "int32",
        "casting": "unsafe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D bool array to int32 with safe casting
    input_dict = {
        "from_": np.array([True, False, True], dtype=np.bool_),
        "to": "int32",
        "casting": "safe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array to int64 with unsafe casting
    input_dict = {
        "from_": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64),
        "to": "int64",
        "casting": "unsafe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array to float32 with equiv casting
    input_dict = {
        "from_": np.zeros((1, 2, 2, 1), dtype=np.float32),
        "to": "float32",
        "casting": "equiv",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array to float32 with no casting
    input_dict = {
        "from_": np.array([0.5, -1.2, 3.4], dtype=np.float32),
        "to": "float32",
        "casting": "no",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int32 array to int32 with no casting
    input_dict = {
        "from_": np.array([[-10, 20], [30, -40]], dtype=np.int32),
        "to": "int32",
        "casting": "no",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float64 array to float64 with equiv casting
    input_dict = {
        "from_": np.random.randn(2, 2, 3).astype(np.float64),
        "to": "float64",
        "casting": "equiv",
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
