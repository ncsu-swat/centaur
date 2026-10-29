
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax.numpy as jnp

class CallableStr(str):
    def __call__(self, *args, **kwargs):
        return getattr(jnp, str(self))(*args, **kwargs)

    def __deepcopy__(self, memo):
        return CallableStr(str(self))

class CustomInt(int):
    def __iter__(self):
        yield int(self)

    def __deepcopy__(self, memo):
        return CustomInt(int(self))

def apply_over_axes_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, func="sum", axis=0
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {
        "func": CallableStr("sum"),
        "a": a,
        "axes": CustomInt(0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, func="min", axis=1
    a = np.array([[7.0, 2.0, 9.0], [4.0, 5.0, 1.0]], dtype=np.float32)
    input_dict = {
        "func": CallableStr("min"),
        "a": a,
        "axes": CustomInt(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array, func="max", axis=2
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {
        "func": CallableStr("max"),
        "a": a,
        "axes": CustomInt(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, func="mean", axis=0
    a = np.random.randn(3, 4, 5).astype(np.float64)
    input_dict = {
        "func": CallableStr("mean"),
        "a": a,
        "axes": CustomInt(0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, func="prod", axis=1
    a = np.random.uniform(0.5, 1.5, size=(2, 2, 3, 2)).astype(np.float32)
    input_dict = {
        "func": CallableStr("prod"),
        "a": a,
        "axes": CustomInt(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array, func="sum", axis=0
    a = np.array([10, -5, 20, -15, 30], dtype=np.int64)
    input_dict = {
        "func": CallableStr("sum"),
        "a": a,
        "axes": CustomInt(0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, func="mean", axis=1
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "func": CallableStr("mean"),
        "a": a,
        "axes": CustomInt(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D bool array, func="all", axis=0
    a = np.array([[True, False, True], [True, True, True]], dtype=bool)
    input_dict = {
        "func": CallableStr("all"),
        "a": a,
        "axes": CustomInt(0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D bool array, func="any", axis=1
    a = np.array([[False, False, False], [False, True, False]], dtype=bool)
    input_dict = {
        "func": CallableStr("any"),
        "a": a,
        "axes": CustomInt(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, func="max", axis=3
    a = np.random.randn(2, 2, 3, 2).astype(np.float32)
    input_dict = {
        "func": CallableStr("max"),
        "a": a,
        "axes": CustomInt(3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.apply_over_axes_3"] = apply_over_axes_inputs()

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
