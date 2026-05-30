
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class CallableStr(str):
    def __new__(cls, name, func):
        obj = super().__new__(cls, name)
        obj.func = func
        return obj
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    def __deepcopy__(self, memo):
        return self
    def __copy__(self):
        return self

def apply_over_axes_inputs():
    list_of_inputs = []

    # Input 1: 2D array, sum over axis 0
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "func": CallableStr("sum", lambda x, axis: x.sum(axis=axis)),
        "a": a,
        "axes": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array, mean over axes 0 and 2
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {
        "func": CallableStr("mean", lambda x, axis: x.mean(axis=axis)),
        "a": a,
        "axes": (0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, max over axis 1
    a = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    input_dict = {
        "func": CallableStr("max", lambda x, axis: x.max(axis=axis)),
        "a": a,
        "axes": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D random array, min over axes 0 and 1
    a = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "func": CallableStr("min", lambda x, axis: x.min(axis=axis)),
        "a": a,
        "axes": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 array, prod over axis 2
    a = np.ones((2, 2, 2), dtype=np.int32) * 2
    input_dict = {
        "func": CallableStr("prod", lambda x, axis: x.prod(axis=axis)),
        "a": a,
        "axes": (2,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D random float32 array, std over axis 0
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "func": CallableStr("std", lambda x, axis: x.std(axis=axis)),
        "a": a,
        "axes": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 array, var over axes 1 and 2
    a = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict = {
        "func": CallableStr("var", lambda x, axis: x.var(axis=axis)),
        "a": a,
        "axes": (1, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int64 array, sum over all axes (0, 1, 2)
    a = np.arange(24).reshape(2, 3, 4).astype(np.int64)
    input_dict = {
        "func": CallableStr("sum", lambda x, axis: x.sum(axis=axis)),
        "a": a,
        "axes": (0, 1, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int32 array, max over axes 0 and 2
    a = np.random.randint(-10, 10, size=(4, 4, 4)).astype(np.int32)
    input_dict = {
        "func": CallableStr("max", lambda x, axis: x.max(axis=axis)),
        "a": a,
        "axes": (0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array, mean over axis 0
    a = np.random.randn(10).astype(np.float32)
    input_dict = {
        "func": CallableStr("mean", lambda x, axis: x.mean(axis=axis)),
        "a": a,
        "axes": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.apply_over_axes_2"] = apply_over_axes_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.apply_over_axes_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.apply_over_axes_2'.")


check_valid('jax.numpy.apply_over_axes', generated_inputs['jax.numpy.apply_over_axes_2'], lib="jax", suffix=2)
