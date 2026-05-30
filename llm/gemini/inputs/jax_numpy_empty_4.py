
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax

# Bypassing JAX PJRT plugin discovery to completely prevent CUDA plugin loading/initialization errors
try:
    import jax._src.xla_bridge as xb
    xb._pjrt_plugins_discovered = True
except Exception:
    pass

# Ensure only CPU platform is registered and active
try:
    jax.config.update("jax_platforms", "cpu")
except Exception:
    pass

from jax.sharding import Mesh, PartitionSpec

try:
    devices = jax.devices()
    mesh = Mesh(devices, ('x',))
    jax._src.mesh.thread_local_state.mesh = mesh
except Exception:
    pass

def empty_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "shape": 4,
        "dtype": "float32",
        "out_sharding": PartitionSpec(None)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "shape": 10,
        "dtype": "int32",
        "out_sharding": PartitionSpec("x")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "shape": 1,
        "dtype": "bool",
        "out_sharding": PartitionSpec(None)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "shape": 100,
        "dtype": "float64",
        "out_sharding": PartitionSpec("x")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "shape": 8,
        "dtype": "uint8",
        "out_sharding": PartitionSpec("x")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "shape": 16,
        "dtype": "int16",
        "out_sharding": PartitionSpec(None)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "shape": 32,
        "dtype": "float16",
        "out_sharding": PartitionSpec("x")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "shape": 64,
        "dtype": "int64",
        "out_sharding": PartitionSpec("x")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "shape": 2,
        "dtype": "bool",
        "out_sharding": PartitionSpec(None)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "shape": 128,
        "dtype": "float32",
        "out_sharding": PartitionSpec("x")
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
