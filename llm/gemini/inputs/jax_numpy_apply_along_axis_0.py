
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp

class CallableStr(str):
    def __call__(self, x, *args, **kwargs):
        if self == "sum":
            return jnp.sum(x, *args, **kwargs)
        elif self == "mean":
            return jnp.mean(x, *args, **kwargs)
        elif self == "std":
            return jnp.std(x, *args, **kwargs)
        elif self == "min":
            return jnp.min(x, *args, **kwargs)
        elif self == "max":
            return jnp.max(x, *args, **kwargs)
        elif self == "ptp":
            return jnp.ptp(x, *args, **kwargs)
        elif self == "argmin":
            return jnp.argmin(x, *args, **kwargs)
        elif self == "argmax":
            return jnp.argmax(x, *args, **kwargs)
        elif self == "prod":
            return jnp.prod(x, *args, **kwargs)
        else:
            return jnp.sum(x, *args, **kwargs)

def apply_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 2D array, sum along axis 0
    input_dict = {
        "func1d": CallableStr("sum"),
        "axis": 0,
        "arr": np.random.randn(5, 5).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, mean along axis 1
    input_dict = {
        "func1d": CallableStr("mean"),
        "axis": 1,
        "arr": np.random.randn(4, 6).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, std along negative axis -1
    input_dict = {
        "func1d": CallableStr("std"),
        "axis": -1,
        "arr": np.random.randn(3, 4, 5).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, min along axis 2
    input_dict = {
        "func1d": CallableStr("min"),
        "axis": 2,
        "arr": np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, max along axis 0
    input_dict = {
        "func1d": CallableStr("max"),
        "axis": 0,
        "arr": np.random.randn(2, 2, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, ptp along axis -2
    input_dict = {
        "func1d": CallableStr("ptp"),
        "axis": -2,
        "arr": np.random.randint(0, 100, size=(3, 3, 3)).astype(np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, argmin along axis 1
    input_dict = {
        "func1d": CallableStr("argmin"),
        "axis": 1,
        "arr": np.random.randn(10, 5).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, argmax along axis 0
    input_dict = {
        "func1d": CallableStr("argmax"),
        "axis": 0,
        "arr": np.random.randint(-50, 50, size=(20,)).astype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with float16, sum along axis 0
    input_dict = {
        "func1d": CallableStr("sum"),
        "axis": 0,
        "arr": np.random.randn(8, 8).astype(np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, prod along axis -1
    input_dict = {
        "func1d": CallableStr("prod"),
        "axis": -1,
        "arr": np.random.randn(2, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.apply_along_axis"] = apply_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.apply_along_axis' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.apply_along_axis'.")


check_valid('jax.numpy.apply_along_axis', generated_inputs['jax.numpy.apply_along_axis'], lib="jax", suffix=0)
