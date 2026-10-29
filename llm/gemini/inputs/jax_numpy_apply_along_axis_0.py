
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp
import numpy as np


class CallableStr(str):

    def __new__(cls, value="sum"):
        return super().__new__(cls, value)

    def __call__(self, arr, *args, **kwargs):
        fn = getattr(jnp, str(self), jnp.sum)
        return fn(arr, *args, **kwargs)


def apply_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"func1d": CallableStr("sum"), "axis": 0, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, negative axis -1
    arr = np.array([-10, 20, -30, 40], dtype=np.int32)
    input_dict = {"func1d": CallableStr("sum"), "axis": -1, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, axis 0
    arr = np.random.randn(4, 5).astype(np.float32)
    input_dict = {"func1d": CallableStr("mean"), "axis": 0, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, axis 1
    arr = np.random.randn(3, 6).astype(np.float64)
    input_dict = {"func1d": CallableStr("sum"), "axis": 1, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int64 array, negative axis -2
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    input_dict = {"func1d": CallableStr("prod"), "axis": -2, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array with negative values, axis -1
    arr = np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0
    input_dict = {"func1d": CallableStr("sum"), "axis": -1, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, axis 0
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"func1d": CallableStr("sum"), "axis": 0, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 array, axis 1
    arr = np.random.randint(-10, 10, size=(3, 4, 5), dtype=np.int32)
    input_dict = {"func1d": CallableStr("prod"), "axis": 1, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 array, axis 2
    arr = np.random.randn(2, 4, 3).astype(np.float64)
    input_dict = {"func1d": CallableStr("mean"), "axis": 2, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, negative axis -1
    arr = np.random.randn(4, 2, 3).astype(np.float32)
    input_dict = {"func1d": CallableStr("sum"), "axis": -1, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D float32 array, axis 2
    arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"func1d": CallableStr("sum"), "axis": 2, "arr": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 5D float32 array, negative axis -3
    arr = np.random.randn(2, 2, 3, 2, 4).astype(np.float32)
    input_dict = {"func1d": CallableStr("mean"), "axis": -3, "arr": arr}
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
