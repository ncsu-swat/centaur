
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, expand at index 0
    array = np.random.randn(5).astype(np.float32)
    dimensions = [0]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    # Input 2: 2D float32 array, expand at indices 0 and 3
    array = np.random.randn(2, 3).astype(np.float32)
    dimensions = [0, 3]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    # Input 3: Scalar (0D array), expand at index 0
    array = np.array(42.0, dtype=np.float32)
    dimensions = [0]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    # Input 4: 2D int32 array, expand at indices 1 and 2
    array = np.random.randint(0, 10, size=(4, 5)).astype(np.int32)
    dimensions = [1, 2]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    # Input 5: 3D float64 array, expand at multiple indices
    array = np.random.randn(2, 2, 2).astype(np.float64)
    dimensions = [0, 2, 4]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    # Input 6: 1D float32 array, multiple contiguous expansions
    array = np.random.randn(3).astype(np.float32)
    dimensions = [0, 1, 2]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    # Input 7: 2D boolean array, expand at index 1
    array = np.random.choice([True, False], size=(3, 3)).astype(np.bool_)
    dimensions = [1]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    # Input 8: 4D float32 array, expand at boundary indices
    array = np.random.randn(1, 2, 3, 4).astype(np.float32)
    dimensions = [0, 5]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    # Input 9: 2D float32 array, expand at a single index in the middle
    array = np.random.randn(10, 10).astype(np.float32)
    dimensions = [1]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    # Input 10: 3D float16 array, expand at multiple indices
    array = np.random.randn(3, 4, 5).astype(np.float16)
    dimensions = [1, 3, 5]
    list_of_inputs.append({"array": copy.deepcopy(array), "dimensions": dimensions})

    return list_of_inputs

generated_inputs["jax.lax.expand_dims_1"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.expand_dims_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.expand_dims_1'.")


check_valid('jax.lax.expand_dims', generated_inputs['jax.lax.expand_dims_1'], lib="jax", suffix=1)
