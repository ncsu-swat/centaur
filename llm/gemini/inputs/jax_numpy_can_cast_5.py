
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def can_cast_inputs():
    list_of_inputs = []

    # Input 1: int32 to int64, safe casting
    list_of_inputs.append({
        "from_": np.array([1, 2, 3], dtype=np.int32),
        "to": np.dtype(np.int64),
        "casting": "safe"
    })

    # Input 2: float64 scalar to float32, same_kind casting
    list_of_inputs.append({
        "from_": np.array(1.5, dtype=np.float64),
        "to": np.dtype(np.float32),
        "casting": "same_kind"
    })

    # Input 3: float32 matrix to complex64, safe casting
    list_of_inputs.append({
        "from_": np.random.randn(2, 2).astype(np.float32),
        "to": np.dtype(np.complex64),
        "casting": "safe"
    })

    # Input 4: bool matrix to int32, safe casting
    list_of_inputs.append({
        "from_": np.array([[True, False]], dtype=np.bool_),
        "to": np.dtype(np.int32),
        "casting": "safe"
    })

    # Input 5: complex128 to float64, unsafe casting
    list_of_inputs.append({
        "from_": np.array([1+2j, 3+4j], dtype=np.complex128),
        "to": np.dtype(np.float64),
        "casting": "unsafe"
    })

    # Input 6: int16 3D tensor to int32, safe casting
    list_of_inputs.append({
        "from_": np.ones((2, 3, 4), dtype=np.int16),
        "to": np.dtype(np.int32),
        "casting": "safe"
    })

    # Input 7: int8 to int8, 'no' casting
    list_of_inputs.append({
        "from_": np.array([10], dtype=np.int8),
        "to": np.dtype(np.int8),
        "casting": "no"
    })

    # Input 8: float64 to float64, 'equiv' casting
    list_of_inputs.append({
        "from_": np.zeros((5,), dtype=np.float64),
        "to": np.dtype(np.float64),
        "casting": "equiv"
    })

    # Input 9: signed int64 to unsigned int64, unsafe casting
    list_of_inputs.append({
        "from_": np.array([-1, 0, 1], dtype=np.int64),
        "to": np.dtype(np.uint64),
        "casting": "unsafe"
    })

    # Input 10: float32 3D tensor to int32, same_kind casting
    list_of_inputs.append({
        "from_": np.random.randn(1, 2, 3).astype(np.float32),
        "to": np.dtype(np.int32),
        "casting": "same_kind"
    })

    return list_of_inputs

generated_inputs["jax.numpy.can_cast_5"] = can_cast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.can_cast_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.can_cast_5'.")


check_valid('jax.numpy.can_cast', generated_inputs['jax.numpy.can_cast_5'], lib="jax", suffix=5)
