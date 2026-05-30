
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp
import jax._src.numpy.util as jax_util

# Monkeypatch JAX to allow tuple inputs and convert them safely inside JAX traces
orig_ensure_arraylike = jax_util.ensure_arraylike

def new_ensure_arraylike(fun_name, *args):
    new_args = tuple(jnp.asarray(x) if isinstance(x, tuple) else x for x in args)
    return orig_ensure_arraylike(fun_name, *new_args)

jax_util.ensure_arraylike = new_ensure_arraylike

def ediff1d_inputs():
    list_of_inputs = []

    # Input 1: Basic integers with positive difference
    input_dict = {
        "ary": (1, 2, 3, 5, 8),
        "to_end": np.array([13, 21], dtype=np.int32),
        "to_begin": np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floating point values
    input_dict = {
        "ary": (1.5, 2.5, 4.0),
        "to_end": np.array([5.5, 7.0], dtype=np.float32),
        "to_begin": np.array([0.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer values
    input_dict = {
        "ary": (-1, -2, -3),
        "to_end": np.array([-4, -5], dtype=np.int32),
        "to_begin": np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Nested 2D-like tuple
    input_dict = {
        "ary": ((1, 2), (3, 4)),
        "to_end": np.array([5, 6], dtype=np.int32),
        "to_begin": np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element tuple
    input_dict = {
        "ary": (10,),
        "to_end": np.array([20], dtype=np.int32),
        "to_begin": np.array([5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Alternating signs
    input_dict = {
        "ary": (1, -1, 1, -1),
        "to_end": np.array([2, -2], dtype=np.int32),
        "to_begin": np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 precision values
    input_dict = {
        "ary": (0.1, 0.2, 0.3),
        "to_end": np.array([0.4], dtype=np.float64),
        "to_begin": np.array([0.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Nested 3D-like tuple
    input_dict = {
        "ary": (((1, 2), (3, 4)), ((5, 6), (7, 8))),
        "to_end": np.array([9, 10], dtype=np.int32),
        "to_begin": np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integers with int64
    input_dict = {
        "ary": (100, 200, 300),
        "to_end": np.array([400, 500], dtype=np.int64),
        "to_begin": np.array([0], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Simple float values
    input_dict = {
        "ary": (10.0, 20.0),
        "to_end": np.array([30.0], dtype=np.float32),
        "to_begin": np.array([0.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ediff1d_5"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_5'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_5'], lib="jax", suffix=5)
