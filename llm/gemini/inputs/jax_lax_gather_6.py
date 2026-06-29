
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
from jax.lax import GatherDimensionNumbers

def gather_inputs():
    list_of_inputs = []
    
    modes = ['clip', 'fill', 'drop', 'promise_in_bounds', 'clip', 'fill', 'drop', 'promise_in_bounds', 'clip', 'fill']
    unique_flags = [False, True, False, True, False, True, False, True, False, False]
    sorted_flags = [False, False, True, True, False, False, True, True, False, False]
    fill_flags = [False, True, False, True, False, True, False, True, False, True]

    for i in range(10):
        operand = np.random.randn(5, 5).astype(np.float32)
        start_indices = np.random.randint(0, 5, size=(3, 1)).astype(np.int32)
        if sorted_flags[i]:
            start_indices = np.sort(start_indices, axis=0)

        dimension_numbers = GatherDimensionNumbers(
            offset_dims=(0,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        )
        
        input_dict = {
            'operand': operand,
            'start_indices': start_indices,
            'dimension_numbers': dimension_numbers,
            'slice_sizes': [1, 5],
            'unique_indices': unique_flags[i],
            'indices_are_sorted': sorted_flags[i],
            'mode': modes[i],
            'fill_value': fill_flags[i]
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_6"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_6'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_6'], lib="jax", suffix=6)
