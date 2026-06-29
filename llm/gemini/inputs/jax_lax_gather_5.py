
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
from jax.lax import GatherDimensionNumbers

def gather_inputs():
    list_of_inputs = []

    # Using the exact same configuration to avoid XLA recompilation overhead.
    dimension_numbers = GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = [1, 2]
    unique_indices = False
    indices_are_sorted = False
    mode = 'clip'
    fill_value = 0

    start_indices_list = [
        [[0], [1]],
        [[1], [2]],
        [[2], [3]],
        [[0], [2]],
        [[1], [3]]
    ]

    for indices_data in start_indices_list:
        operand = np.random.randn(4, 2).astype(np.float32)
        start_indices = np.array(indices_data, dtype=np.int32)
        input_dict = {
            'operand': operand,
            'start_indices': start_indices,
            'dimension_numbers': dimension_numbers,
            'slice_sizes': slice_sizes,
            'unique_indices': unique_indices,
            'indices_are_sorted': indices_are_sorted,
            'mode': mode,
            'fill_value': fill_value
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_5"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_5'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_5'], lib="jax", suffix=5)
