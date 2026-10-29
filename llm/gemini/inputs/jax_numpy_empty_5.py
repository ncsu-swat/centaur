
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

# Set up an active mesh context required when out_sharding is specified as a PartitionSpec
_mesh = Mesh(np.array(jax.devices()), ("axis",))
_mesh_ctx = jax.set_mesh(_mesh)
_mesh_ctx.__enter__()


def empty_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar, float32
    input_dict = {
        "shape": (),
        "dtype": "float32",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-D array, int32
    input_dict = {
        "shape": (8,),
        "dtype": "int32",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2-D array, bool
    input_dict = {
        "shape": (4, 6),
        "dtype": "bool",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2-D square matrix, float64
    input_dict = {
        "shape": (5, 5),
        "dtype": "float64",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3-D array, complex64
    input_dict = {
        "shape": (2, 3, 4),
        "dtype": "complex64",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1-D zero-size array, uint8
    input_dict = {
        "shape": (0,),
        "dtype": "uint8",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3-D array with zero dimension, int8
    input_dict = {
        "shape": (3, 0, 2),
        "dtype": "int8",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4-D array, float16
    input_dict = {
        "shape": (2, 2, 2, 2),
        "dtype": "float16",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2-D row vector, int64
    input_dict = {
        "shape": (1, 10),
        "dtype": "int64",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2-D column vector, uint32
    input_dict = {
        "shape": (10, 1),
        "dtype": "uint32",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4-D singleton shape, bfloat16
    input_dict = {
        "shape": (1, 1, 1, 1),
        "dtype": "bfloat16",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 3-D large array, complex128
    input_dict = {
        "shape": (4, 8, 16),
        "dtype": "complex128",
        "out_sharding": P(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.empty_5"] = empty_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_5'.")


check_valid('jax.numpy.empty', generated_inputs['jax.numpy.empty_5'], lib="jax", suffix=5)
