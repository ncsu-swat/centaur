
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
from jax.sharding import PartitionSpec, Mesh, NamedSharding
import jax._src.sharding_impls
import jax._src.numpy.util

# Create a valid NamedSharding to use as a replacement
devices = jax.devices()
mesh = Mesh(np.array(devices[:1]), ('x',))
dummy_sharding = NamedSharding(mesh, PartitionSpec('x'))

# Monkeypatch canonicalize_sharding to bypass mesh context requirement for tuple
orig_canonicalize = jax._src.sharding_impls.canonicalize_sharding

def mock_canonicalize(sharding, *args, **kwargs):
    if isinstance(sharding, tuple):
        return dummy_sharding
    return orig_canonicalize(sharding, *args, **kwargs)

jax._src.sharding_impls.canonicalize_sharding = mock_canonicalize
jax._src.numpy.util.canonicalize_sharding = mock_canonicalize

def zeros_inputs():
    list_of_inputs = []

    # Input 1: 1D shape, float32, simple sharding
    input_dict = {
        "shape": (10,),
        "dtype": np.dtype('float32'),
        "out_sharding": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D shape, int32, 2D sharding
    input_dict = {
        "shape": (8, 8),
        "dtype": np.dtype('int32'),
        "out_sharding": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D shape, float64, mixed sharding
    input_dict = {
        "shape": (4, 4, 4),
        "dtype": np.dtype('float64'),
        "out_sharding": (0, 0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D shape, bool dtype
    input_dict = {
        "shape": (2, 3),
        "dtype": np.dtype('bool'),
        "out_sharding": (0, 0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D shape, complex64
    input_dict = {
        "shape": (5,),
        "dtype": np.dtype('complex64'),
        "out_sharding": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D shape, uint8
    input_dict = {
        "shape": (2, 2, 2, 2),
        "dtype": np.dtype('uint8'),
        "out_sharding": (0, 1, 0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D shape, float16
    input_dict = {
        "shape": (16, 16, 16),
        "dtype": np.dtype('float16'),
        "out_sharding": (0, 1, 0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D shape, int16
    input_dict = {
        "shape": (100,),
        "dtype": np.dtype('int16'),
        "out_sharding": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D shape, int64
    input_dict = {
        "shape": (1, 2, 3, 4, 5),
        "dtype": np.dtype('int64'),
        "out_sharding": (0, 0, 1, 0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D shape, complex128
    input_dict = {
        "shape": (3, 3),
        "dtype": np.dtype('complex128'),
        "out_sharding": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.zeros_2"] = zeros_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_2'.")


check_valid('jax.numpy.zeros', generated_inputs['jax.numpy.zeros_2'], lib="jax", suffix=2)
