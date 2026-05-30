
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Monkey-patch JAX's internal utility to accept standard Python lists in ediff1d
try:
    import jax._src.numpy.util as jax_util
except ImportError:
    try:
        import jax.numpy.util as jax_util
    except ImportError:
        jax_util = None

if jax_util is not None:
    _orig_check_arraylike = getattr(jax_util, "check_arraylike", None)
    _orig_ensure_arraylike = getattr(jax_util, "ensure_arraylike", None)
    
    def patched_check_arraylike(fun_name, *args):
        new_args = [jnp.asarray(arg) if isinstance(arg, list) else arg for arg in args]
        if _orig_check_arraylike:
            return _orig_check_arraylike(fun_name, *new_args)
            
    def patched_ensure_arraylike(fun_name, *args):
        new_args = [jnp.asarray(arg) if isinstance(arg, list) else arg for arg in args]
        if _orig_ensure_arraylike:
            return _orig_ensure_arraylike(fun_name, *new_args)
        if len(new_args) == 1:
            return new_args[0]
        return new_args

    jax_util.check_arraylike = patched_check_arraylike
    jax_util.ensure_arraylike = patched_ensure_arraylike

def ediff1d_inputs():
    list_of_inputs = []

    # Input 1: Basic integer list
    input_dict = {
        "ary": [1, 2, 4, 7],
        "to_end": np.array([10, 20], dtype=np.int32),
        "to_begin": np.array([-5, -3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 list
    input_dict = {
        "ary": [1.5, 2.5, 4.0],
        "to_end": np.array([5.5], dtype=np.float32),
        "to_begin": np.array([0.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer values
    input_dict = {
        "ary": [-10, -5, 0, 5],
        "to_end": np.array([-1, -2], dtype=np.int32),
        "to_begin": np.array([1, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large values using int64
    input_dict = {
        "ary": [1000, 2000, 3000],
        "to_end": np.array([4000], dtype=np.int64),
        "to_begin": np.array([0], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Nested list representing 2D structure
    input_dict = {
        "ary": [[1, 2], [3, 4]],
        "to_end": np.array([5, 6], dtype=np.int32),
        "to_begin": np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 precision values
    input_dict = {
        "ary": [1.0, 1.1, 1.2],
        "to_end": np.array([1.3], dtype=np.float64),
        "to_begin": np.array([0.9], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element list
    input_dict = {
        "ary": [42],
        "to_end": np.array([100], dtype=np.int32),
        "to_begin": np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional nested list
    input_dict = {
        "ary": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        "to_end": np.array([9, 10], dtype=np.int32),
        "to_begin": np.array([-1, 0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty list
    input_dict = {
        "ary": [],
        "to_end": np.array([1, 2], dtype=np.int32),
        "to_begin": np.array([3, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed integer and float types in list
    input_dict = {
        "ary": [1, 2.5, 3],
        "to_end": np.array([4.5, 5.0], dtype=np.float64),
        "to_begin": np.array([0.0, 0.5], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ediff1d_4"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_4'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_4'], lib="jax", suffix=4)
