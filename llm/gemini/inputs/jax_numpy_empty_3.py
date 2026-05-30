
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Initialize a global mesh context on CPU to allow PartitionSpec usage via jax.set_mesh
devices = np.array(jax.devices("cpu")[:1])
mesh = jax.sharding.Mesh(devices, ('x',))
context_manager = jax.set_mesh(mesh)
context_manager.__enter__()

def empty_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "shape": [4],
        "dtype": np.dtype('float32'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "shape": [2, 3],
        "dtype": np.dtype('int32'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "shape": [8, 8],
        "dtype": np.dtype('bool'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "shape": [1, 10, 10],
        "dtype": np.dtype('float64'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "shape": [100],
        "dtype": np.dtype('int64'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "shape": [3, 3, 3],
        "dtype": np.dtype('complex64'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "shape": [2, 2, 2, 2],
        "dtype": np.dtype('int8'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "shape": [5, 12],
        "dtype": np.dtype('uint8'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "shape": [10, 20],
        "dtype": np.dtype('float16'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "shape": [16],
        "dtype": np.dtype('uint32'),
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.empty_3"] = empty_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_3'.")


check_valid('jax.numpy.empty', generated_inputs['jax.numpy.empty_3'], lib="jax", suffix=3)
