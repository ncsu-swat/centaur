
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp
import numpy as np


class CallableStr(str):

    def __call__(self, a, *args, **kwargs):
        fn = getattr(jnp, str(self), None)
        if fn is None:
            fn = getattr(np, str(self))
        return fn(a, *args, **kwargs)


def apply_over_axes_inputs():
    list_of_inputs = []

    # Input 1: 2D integer array, single axis sum
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    axes = [0]
    func = CallableStr("sum")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    # Input 2: 2D float array, single axis min
    a = np.array([[1.5, -2.3, 3.1], [4.0, -0.5, 6.2]], dtype=np.float32)
    axes = [1]
    func = CallableStr("min")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    # Input 3: 2D integer array, multiple axes prod
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    axes = [0, 1]
    func = CallableStr("prod")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    # Input 4: 3D float array, multiple axes max
    a = np.random.randn(2, 4, 3).astype(np.float32)
    axes = [1, 2]
    func = CallableStr("max")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    # Input 5: 3D float64 array with negative values, axes 0 and 2 sum
    a = np.random.uniform(-10.0, 10.0, size=(3, 3, 3)).astype(np.float64)
    axes = [0, 2]
    func = CallableStr("sum")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    # Input 6: 1D float array, axis 0 mean
    a = np.linspace(-5.0, 5.0, 10, dtype=np.float32)
    axes = [0]
    func = CallableStr("mean")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    # Input 7: 4D float array, axes 0 and 3 mean
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = [0, 3]
    func = CallableStr("mean")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    # Input 8: 3D float32 array, axis 1 prod
    a = np.random.uniform(0.5, 2.0, size=(2, 3, 2)).astype(np.float32)
    axes = [1]
    func = CallableStr("prod")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    # Input 9: 4D integer array, axis 2 prod
    a = np.random.randint(1, 5, size=(2, 2, 3, 2), dtype=np.int32)
    axes = [2]
    func = CallableStr("prod")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    # Input 10: 3D float64 array, axes 0 and 1 min
    a = np.random.randn(4, 3, 2).astype(np.float64)
    axes = [0, 1]
    func = CallableStr("min")
    list_of_inputs.append(copy.deepcopy({"func": func, "a": a, "axes": axes}))

    return list_of_inputs


generated_inputs["jax.numpy.apply_over_axes_1"] = apply_over_axes_inputs()

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
