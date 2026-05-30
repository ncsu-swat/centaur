
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class CallableStr(str):
    def __call__(self, x, axis):
        func_map = {
            "sum": np.sum,
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "prod": np.prod,
            "std": np.std,
            "var": np.var,
        }
        fn = func_map.get(self, np.sum)
        return fn(x, axis=axis)

class IterableInt(int):
    def __iter__(self):
        return iter([int(self)])

def jax_numpy_apply_over_axes_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 array, sum over axis 0
    input_dict = {
        "func": CallableStr("sum"),
        "a": np.arange(24).reshape(2, 3, 4).astype(np.float32),
        "axes": IterableInt(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array, mean over axis 1
    input_dict = {
        "func": CallableStr("mean"),
        "a": np.random.randn(5, 5).astype(np.float64),
        "axes": IterableInt(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array with negative values, max over axis 2
    input_dict = {
        "func": CallableStr("max"),
        "a": np.random.randint(-10, 10, size=(3, 3, 3)).astype(np.int32),
        "axes": IterableInt(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float32 array, min over axis 1
    input_dict = {
        "func": CallableStr("min"),
        "a": np.random.uniform(-5.0, 5.0, size=(2, 3, 2, 2)).astype(np.float32),
        "axes": IterableInt(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int64 array, prod over axis 0
    input_dict = {
        "func": CallableStr("prod"),
        "a": np.random.randint(1, 5, size=(4, 3)).astype(np.int64),
        "axes": IterableInt(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, std over axis 2
    input_dict = {
        "func": CallableStr("std"),
        "a": np.random.randn(2, 4, 3).astype(np.float32),
        "axes": IterableInt(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array, var over axis 0
    input_dict = {
        "func": CallableStr("var"),
        "a": np.random.randn(6, 6).astype(np.float64),
        "axes": IterableInt(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D float32 array, sum over axis 3
    input_dict = {
        "func": CallableStr("sum"),
        "a": np.ones((2, 2, 2, 3, 2), dtype=np.float32),
        "axes": IterableInt(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 array, max over axis 0
    input_dict = {
        "func": CallableStr("max"),
        "a": np.array([-1.2, 3.4, 5.6, -7.8, 9.0], dtype=np.float32),
        "axes": IterableInt(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float64 array with negative values, min over axis 1
    input_dict = {
        "func": CallableStr("min"),
        "a": np.random.uniform(-10.0, 10.0, size=(3, 2, 4)).astype(np.float64),
        "axes": IterableInt(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.apply_over_axes_3"] = jax_numpy_apply_over_axes_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.apply_over_axes_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.apply_over_axes_3'.")


check_valid('jax.numpy.apply_over_axes', generated_inputs['jax.numpy.apply_over_axes_3'], lib="jax", suffix=3)
