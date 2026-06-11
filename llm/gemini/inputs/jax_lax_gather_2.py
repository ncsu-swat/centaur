
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

def make_dimension_numbers(offset_dims, collapsed_slice_dims, start_index_map):
    values = {
        'offset_dims': offset_dims,
        'collapsed_slice_dims': collapsed_slice_dims,
        'start_index_map': start_index_map
    }
    for field in jax.lax.GatherDimensionNumbers._fields:
        if field not in values:
            values[field] = (0,)
    args = [values[field] for field in jax.lax.GatherDimensionNumbers._fields]
    return jax.lax.GatherDimensionNumbers(*args)

def gather_inputs():
    list_of_inputs = []
    has_5_fields = (len(jax.lax.GatherDimensionNumbers._fields) == 5)

    for i in range(10):
        unique = (i % 2 == 0)
        sorted_idx = (i % 3 == 0)
        mode = ["clip", "promise_in_bounds", "fill"][i % 3]
        fill_val = -1 if i % 2 == 0 else 99
        
        dtypes = [np.int32, np.float32, np.int64, np.float64]
        dtype = dtypes[i % len(dtypes)]

        if has_5_fields:
            operand = np.arange(24, dtype=dtype).reshape(2, 3, 4)
            start_indices = np.array([[[0], [2]], [[1], [2]]], dtype=np.int32)
            dimension_numbers = make_dimension_numbers(
                offset_dims=(2,),
                collapsed_slice_dims=(1,),
                start_index_map=(1,)
            )
            slice_sizes = (1, 1, 2)
        else:
            operand = np.arange(12, dtype=dtype).reshape(3, 4)
            if mode == "fill":
                start_indices = np.array([[0], [5]], dtype=np.int32)
            else:
                start_indices = np.array([[0], [2]], dtype=np.int32)
                
            dimension_numbers = make_dimension_numbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0,),
                start_index_map=(0,)
            )
            slice_sizes = (1, 2)

        input_dict = {
            "operand": operand,
            "start_indices": start_indices,
            "dimension_numbers": dimension_numbers,
            "slice_sizes": slice_sizes,
            "unique_indices": unique,
            "indices_are_sorted": sorted_idx,
            "mode": mode,
            "fill_value": fill_val
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_2"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_2'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_2'], lib="jax", suffix=2)
