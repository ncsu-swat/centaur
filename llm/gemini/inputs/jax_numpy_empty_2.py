
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
from jax.sharding import Mesh, PartitionSpec
import numpy as np

# Ensure an active mesh context for PartitionSpec
_mesh = Mesh(np.array(jax.devices()), ("x",))
_mesh_ctx = jax.set_mesh(_mesh)
_mesh_ctx.__enter__()


def empty_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32
    input_dict = {
        "shape": (8,),
        "dtype": np.dtype(np.float32),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, int32
    input_dict = {
        "shape": (4, 4),
        "dtype": np.dtype(np.int32),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, float64
    input_dict = {
        "shape": (2, 3, 5),
        "dtype": np.dtype(np.float64),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar shape, int8
    input_dict = {
        "shape": (),
        "dtype": np.dtype(np.int8),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty 1D array (size 0), uint8
    input_dict = {
        "shape": (0,),
        "dtype": np.dtype(np.uint8),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array with unit dimensions, bool
    input_dict = {
        "shape": (1, 1, 1, 1),
        "dtype": np.dtype(np.bool_),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, complex64
    input_dict = {
        "shape": (16,),
        "dtype": np.dtype(np.complex64),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, int64
    input_dict = {
        "shape": (3, 7),
        "dtype": np.dtype(np.int64),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with zero dimension, float16
    input_dict = {
        "shape": (2, 0, 3),
        "dtype": np.dtype(np.float16),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, complex128
    input_dict = {
        "shape": (2, 2, 2, 2),
        "dtype": np.dtype(np.complex128),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array, int16
    input_dict = {
        "shape": (5,),
        "dtype": np.dtype(np.int16),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.empty_2"] = empty_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_2'.")


check_valid('jax.numpy.empty', generated_inputs['jax.numpy.empty_2'], lib="jax", suffix=2)
