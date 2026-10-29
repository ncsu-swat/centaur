
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp
import numpy as np


class CallableStr(str):

    def __call__(self, a, axis):
        return getattr(jnp, str(self))(a, axis)

    def __deepcopy__(self, memo):
        return self


def apply_over_axes_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, sum over axis 0
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"func": CallableStr("sum"), "a": a, "axes": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, sum over axis 1
    a = np.array([[-1.0, 2.5, -3.0], [4.5, -5.0, 6.0]], dtype=np.float32)
    input_dict = {"func": CallableStr("sum"), "a": a, "axes": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array, min over axes (0, 1)
    a = np.array([[10, 20, 5], [3, 15, 8]], dtype=np.int32)
    input_dict = {"func": CallableStr("min"), "a": a, "axes": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, max over axis 1
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"func": CallableStr("max"), "a": a, "axes": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 array, prod over axis 0
    a = np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"func": CallableStr("prod"), "a": a, "axes": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, mean over axes (0, 2)
    a = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {"func": CallableStr("mean"), "a": a, "axes": (0, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float64 array, sum over axis 0
    a = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float64)
    input_dict = {"func": CallableStr("sum"), "a": a, "axes": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, max over axes (1, 3)
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    input_dict = {"func": CallableStr("max"), "a": a, "axes": (1, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int64 array, min over axes (0, 1)
    a = np.random.randint(-100, 100, size=(3, 4, 5), dtype=np.int64)
    input_dict = {"func": CallableStr("min"), "a": a, "axes": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, prod over axes (0, 1, 2)
    a = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.0, 0.5]]], dtype=np.float32
    )
    input_dict = {"func": CallableStr("prod"), "a": a, "axes": (0, 1, 2)}
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
