
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import jax.numpy as jnp
import numpy as np


class IndexTuple(tuple):

    @property
    def __class__(self):
        return jax.Array

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return (len(self),)

    @property
    def dtype(self):
        return np.dtype(np.int32)

    def ravel(self, *args, **kwargs):
        return jnp.asarray(tuple(self), dtype=np.int32).ravel(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return jnp.asarray(tuple(self), dtype=np.int32).reshape(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(jnp.asarray(tuple(self), dtype=np.int32), name)


def put_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, basic indices, clip mode
    a = np.zeros(5, dtype=np.int32)
    ind = IndexTuple((0, 2, 4))
    v = np.array([10, 20, 30], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, flattened indices, wrap mode
    a = np.zeros((3, 4), dtype=np.float32)
    ind = IndexTuple((1, 5, 11))
    v = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64 array, out-of-bound indices with clip mode
    a = np.ones(6, dtype=np.float64)
    ind = IndexTuple((0, 7, 10))
    v = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float32 array, out-of-bound indices with wrap mode
    a = np.zeros(5, dtype=np.float32)
    ind = IndexTuple((5, 6, 7))
    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, clip mode
    a = np.zeros((2, 3, 2), dtype=np.int64)
    ind = IndexTuple((0, 3, 11))
    v = np.array([10, 20, 30], dtype=np.int64)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D boolean array, clip mode
    a = np.zeros((3, 3), dtype=np.bool_)
    ind = IndexTuple((0, 4, 8))
    v = np.array([True, True, True], dtype=np.bool_)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D complex64 array, wrap mode
    a = np.zeros(4, dtype=np.complex64)
    ind = IndexTuple((0, 2))
    v = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D int16 array, single-element index tuple, clip mode
    a = np.zeros((2, 2, 2, 2), dtype=np.int16)
    ind = IndexTuple((15,))
    v = np.array([42], dtype=np.int16)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D uint8 array, repeating indices, wrap mode
    a = np.zeros(8, dtype=np.uint8)
    ind = IndexTuple((1, 1, 3, 3))
    v = np.array([5, 10, 15, 20], dtype=np.uint8)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, multiple indices, clip mode
    a = np.ones((2, 3, 4), dtype=np.float32)
    ind = IndexTuple((0, 5, 10, 15, 20))
    v = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.put_3"] = put_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_3'.")


check_valid('jax.numpy.put', generated_inputs['jax.numpy.put_3'], lib="jax", suffix=3)
