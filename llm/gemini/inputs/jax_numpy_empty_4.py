
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
from jax.sharding import Mesh, PartitionSpec
import numpy as np

# Set up active mesh context for PartitionSpec out_sharding
_devices = np.array(jax.devices())
_mesh = Mesh(_devices, ("x",))
_mesh_ctx = jax.set_mesh(_mesh)
_mesh_ctx.__enter__()


def empty_inputs():
    list_of_inputs = []

    # Input 1: shape 0, int32
    input_dict = {
        "shape": 0,
        "dtype": "int32",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: shape 1, float32
    input_dict = {
        "shape": 1,
        "dtype": "float32",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: shape 2, bool
    input_dict = {
        "shape": 2,
        "dtype": "bool",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: shape 4, float16
    input_dict = {
        "shape": 4,
        "dtype": "float16",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: shape 8, bfloat16
    input_dict = {
        "shape": 8,
        "dtype": "bfloat16",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: shape 16, int8
    input_dict = {
        "shape": 16,
        "dtype": "int8",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: shape 3, uint8
    input_dict = {
        "shape": 3,
        "dtype": "uint8",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: shape 5, complex64
    input_dict = {
        "shape": 5,
        "dtype": "complex64",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: shape 6, int16
    input_dict = {
        "shape": 6,
        "dtype": "int16",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: shape 10, uint32
    input_dict = {
        "shape": 10,
        "dtype": "uint32",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: shape 32, float64
    input_dict = {
        "shape": 32,
        "dtype": "float64",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: shape 64, complex128
    input_dict = {
        "shape": 64,
        "dtype": "complex128",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.empty_4"] = empty_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_4'.")


check_valid('jax.numpy.empty', generated_inputs['jax.numpy.empty_4'], lib="jax", suffix=4)
