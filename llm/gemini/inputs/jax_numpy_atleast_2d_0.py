
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atleast_2d_inputs():
    list_of_inputs = []

    # Input 1: 0-D tensor (scalar), float32
    arys = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 2: 1-D tensor, int32
    arys = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 3: 2-D tensor, float64 with negative values
    arys = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 4: 3-D tensor, uint8
    arys = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 5: 4-D tensor, float32 random
    arys = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 6: 1-D tensor, boolean
    arys = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 7: 0-D tensor, int64
    arys = np.array(-100, dtype=np.int64)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 8: 5-D tensor, float16
    arys = np.ones((1, 2, 1, 3, 1), dtype=np.float16)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 9: Empty/zero-sized 2-D tensor
    arys = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 10: 1-D tensor with 1 element, int16
    arys = np.array([42], dtype=np.int16)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    return list_of_inputs

generated_inputs["jax.numpy.atleast_2d"] = atleast_2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_2d'.")


check_valid('jax.numpy.atleast_2d', generated_inputs['jax.numpy.atleast_2d'], lib="jax", suffix=0)
