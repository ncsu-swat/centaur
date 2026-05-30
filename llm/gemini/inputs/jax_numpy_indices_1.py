
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def indices_inputs():
    list_of_inputs = []

    # Input 1: 2D dense grid with int32
    list_of_inputs.append({
        "dimensions": (2, 3),
        "dtype": np.dtype('int32'),
        "sparse": False
    })

    # Input 2: 2D sparse grid with int64
    list_of_inputs.append({
        "dimensions": (3, 4),
        "dtype": np.dtype('int64'),
        "sparse": True
    })

    # Input 3: 1D dense grid with int16
    list_of_inputs.append({
        "dimensions": (5,),
        "dtype": np.dtype('int16'),
        "sparse": False
    })

    # Input 4: 3D sparse grid with int32
    list_of_inputs.append({
        "dimensions": (2, 2, 2),
        "dtype": np.dtype('int32'),
        "sparse": True
    })

    # Input 5: 2D dense grid with unit dimensions
    list_of_inputs.append({
        "dimensions": (1, 5),
        "dtype": np.dtype('uint32'),
        "sparse": False
    })

    # Input 6: 3D dense grid with int64
    list_of_inputs.append({
        "dimensions": (3, 1, 2),
        "dtype": np.dtype('int64'),
        "sparse": False
    })

    # Input 7: 2D sparse grid with int16
    list_of_inputs.append({
        "dimensions": (10, 5),
        "dtype": np.dtype('int16'),
        "sparse": True
    })

    # Input 8: Large 3D dense grid
    list_of_inputs.append({
        "dimensions": (4, 4, 4),
        "dtype": np.dtype('int32'),
        "sparse": False
    })

    # Input 9: 1D sparse grid with int32
    list_of_inputs.append({
        "dimensions": (6,),
        "dtype": np.dtype('int32'),
        "sparse": True
    })

    # Input 10: 4D sparse grid with int64
    list_of_inputs.append({
        "dimensions": (2, 3, 2, 3),
        "dtype": np.dtype('int64'),
        "sparse": True
    })

    # Input 11: 2D dense grid with float32 (valid dtype for indexing in some contexts)
    list_of_inputs.append({
        "dimensions": (3, 3),
        "dtype": np.dtype('float32'),
        "sparse": False
    })

    return list_of_inputs

generated_inputs["jax.numpy.indices_1"] = indices_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.indices_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.indices_1'.")


check_valid('jax.numpy.indices', generated_inputs['jax.numpy.indices_1'], lib="jax", suffix=1)
