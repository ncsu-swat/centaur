
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax
import numpy as np
import copy
from jax.sharding import Mesh, PartitionSpec

# Set up global mesh context to allow PartitionSpec usage
devices = jax.devices()
mesh = Mesh(np.array(devices[:1]), ('x',))
jax.set_mesh(mesh)

def empty_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "shape": 10,
        "dtype": "float32",
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "shape": 5,
        "dtype": "int32",
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "shape": 100,
        "dtype": "bool",
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "shape": 1,
        "dtype": "float64",
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "shape": 256,
        "dtype": "int64",
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "shape": 3,
        "dtype": "complex64",
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "shape": 1024,
        "dtype": "uint8",
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "shape": 8,
        "dtype": "float16",
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "shape": 16,
        "dtype": "int16",
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "shape": 32,
        "dtype": "uint32",
        "out_sharding": PartitionSpec()
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
