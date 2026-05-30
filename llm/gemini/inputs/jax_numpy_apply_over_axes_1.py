
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp

class CallableStr(str):
    def __new__(cls, value, func):
        obj = super().__new__(cls, value)
        obj.func = func
        return obj
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    def __reduce__(self):
        return (CallableStr, (str(self), self.func))

def jax_numpy_apply_over_axes_inputs():
    list_of_inputs = []

    # Input 1: 2D array, sum over axis 0
    input_dict = {
        "func": CallableStr("sum", jnp.sum),
        "a": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "axes": [0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float32 array, mean over axes 0 and 2
    input_dict = {
        "func": CallableStr("mean", jnp.mean),
        "a": np.random.randn(2, 3, 4).astype(np.float32),
        "axes": [0, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, max over axis 1
    input_dict = {
        "func": CallableStr("max", jnp.max),
        "a": np.arange(24).reshape(2, 3, 4).astype(np.float64),
        "axes": [1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array with negative values, min over axes 0 and 1
    input_dict = {
        "func": CallableStr("min", jnp.min),
        "a": np.random.randint(-10, 10, size=(5, 5)).astype(np.int64),
        "axes": [0, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, product over axis 2
    input_dict = {
        "func": CallableStr("prod", jnp.prod),
        "a": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        "axes": [2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, std over axes 1 and 3
    input_dict = {
        "func": CallableStr("std", jnp.std),
        "a": np.random.uniform(0.1, 1.0, size=(2, 2, 2, 2)).astype(np.float32),
        "axes": [1, 3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array, variance over axis 0
    input_dict = {
        "func": CallableStr("var", jnp.var),
        "a": np.random.randn(3, 3).astype(np.float64),
        "axes": [0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D boolean array, logical AND (all) over axis 1
    input_dict = {
        "func": CallableStr("all", jnp.all),
        "a": np.array([[True, False], [True, True]], dtype=bool),
        "axes": [1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D boolean array, logical OR (any) over axes 0 and 2
    input_dict = {
        "func": CallableStr("any", jnp.any),
        "a": np.array([[[False, False], [True, False]]], dtype=bool),
        "axes": [0, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array, sum over axis 0
    input_dict = {
        "func": CallableStr("sum", jnp.sum),
        "a": np.ones((10,), dtype=np.float32),
        "axes": [0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.apply_over_axes_1"] = jax_numpy_apply_over_axes_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.apply_over_axes_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.apply_over_axes_1'.")


check_valid('jax.numpy.apply_over_axes', generated_inputs['jax.numpy.apply_over_axes_1'], lib="jax", suffix=1)
