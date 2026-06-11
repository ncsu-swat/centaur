
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class FakeGatherDimensionNumbers:
    _fields = ('offset_dims', 'collapsed_slice_dims', 'operand_batching_dims', 'start_indices_batching_dims', 'start_index_map')
    def __init__(self, offset_dims, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, start_index_map):
        self.offset_dims = offset_dims
        self.collapsed_slice_dims = collapsed_slice_dims
        self.operand_batching_dims = operand_batching_dims
        self.start_indices_batching_dims = start_indices_batching_dims
        self.start_index_map = start_index_map
        self._data = (offset_dims, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, start_index_map)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

def gather_inputs():
    list_of_inputs = []

    # Case 1: 3D operand (all 5 fields of length 1)
    list_of_inputs.append({
        "operand": np.arange(50, dtype=np.float32).reshape(2, 5, 5),
        "start_indices": np.array([[[1]], [[3]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(2,),
            collapsed_slice_dims=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
            start_index_map=(1,)
        ),
        "slice_sizes": [1, 1, 5],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0.0
    })

    # Case 2: 3D operand, different values
    list_of_inputs.append({
        "operand": np.arange(48, dtype=np.float32).reshape(3, 4, 4),
        "start_indices": np.array([[[1], [2]], [[0], [1]], [[2], [0]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(2,),
            collapsed_slice_dims=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
            start_index_map=(1,)
        ),
        "slice_sizes": [1, 1, 4],
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "fill",
        "fill_value": -1.0
    })

    # Case 3: 3D operand, mapping different dim
    list_of_inputs.append({
        "operand": np.arange(72, dtype=np.float32).reshape(2, 6, 6),
        "start_indices": np.array([[[1], [2], [0], [3]], [[2], [0], [1], [2]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(2,),
            collapsed_slice_dims=(2,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
            start_index_map=(2,)
        ),
        "slice_sizes": [1, 6, 1],
        "unique_indices": False,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": 0.0
    })

    # Case 4: 3D operand, unique_indices
    list_of_inputs.append({
        "operand": np.arange(100, dtype=np.float32).reshape(4, 5, 5),
        "start_indices": np.array([[[1], [2]], [[0], [1]], [[2], [0]], [[1], [2]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(2,),
            collapsed_slice_dims=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
            start_index_map=(1,)
        ),
        "slice_sizes": [1, 1, 5],
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 99.0
    })

    # Case 5: 3D operand, larger index shape
    list_of_inputs.append({
        "operand": np.arange(128, dtype=np.float32).reshape(2, 8, 8),
        "start_indices": np.array([[[1], [2], [3], [4], [0]], [[0], [1], [2], [3], [4]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(2,),
            collapsed_slice_dims=(2,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
            start_index_map=(2,)
        ),
        "slice_sizes": [1, 8, 1],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": -99.0
    })

    # Case 6: 6D operand (all 5 fields of length 2)
    list_of_inputs.append({
        "operand": np.arange(1024, dtype=np.float32).reshape(2, 2, 4, 4, 4, 4),
        "start_indices": np.array([[[[0, 0], [1, 1], [2, 2]], [[1, 1], [2, 2], [0, 0]]],
                                   [[[2, 2], [0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(3, 4),
            collapsed_slice_dims=(2, 3),
            operand_batching_dims=(0, 1),
            start_indices_batching_dims=(0, 1),
            start_index_map=(2, 3)
        ),
        "slice_sizes": [1, 1, 1, 1, 4, 4],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0.0
    })

    # Case 7: 6D operand, unique_indices
    list_of_inputs.append({
        "operand": np.arange(2500, dtype=np.float32).reshape(2, 2, 5, 5, 5, 5),
        "start_indices": np.array([[[[0, 0], [1, 1]], [[2, 2], [3, 3]]],
                                   [[[3, 3], [2, 2]], [[1, 1], [0, 0]]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(3, 4),
            collapsed_slice_dims=(2, 3),
            operand_batching_dims=(0, 1),
            start_indices_batching_dims=(0, 1),
            start_index_map=(2, 3)
        ),
        "slice_sizes": [1, 1, 1, 1, 5, 5],
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "fill",
        "fill_value": -1.5
    })

    # Case 8: 6D operand, promise_in_bounds
    list_of_inputs.append({
        "operand": np.arange(324, dtype=np.float32).reshape(2, 2, 3, 3, 3, 3),
        "start_indices": np.array([[[[0, 0]], [[1, 1]]],
                                   [[[1, 1]], [[0, 0]]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(3, 4),
            collapsed_slice_dims=(2, 3),
            operand_batching_dims=(0, 1),
            start_indices_batching_dims=(0, 1),
            start_index_map=(2, 3)
        ),
        "slice_sizes": [1, 1, 1, 1, 3, 3],
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": 0.0
    })

    # Case 9: 6D operand, different dimension sizes
    list_of_inputs.append({
        "operand": np.arange(1600, dtype=np.float32).reshape(2, 2, 4, 4, 5, 5),
        "start_indices": np.array([[[[0, 0], [1, 1]], [[1, 1], [0, 0]]],
                                   [[[0, 0], [1, 1]], [[1, 1], [0, 0]]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(3, 4),
            collapsed_slice_dims=(2, 3),
            operand_batching_dims=(0, 1),
            start_indices_batching_dims=(0, 1),
            start_index_map=(2, 3)
        ),
        "slice_sizes": [1, 1, 1, 1, 5, 5],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 2.5
    })

    # Case 10: 6D operand, negative fill value
    list_of_inputs.append({
        "operand": np.arange(1024, dtype=np.float32).reshape(2, 2, 4, 4, 4, 4),
        "start_indices": np.array([[[[0, 0], [1, 1]], [[1, 1], [0, 0]]],
                                   [[[0, 0], [1, 1]], [[1, 1], [0, 0]]]], dtype=np.int32),
        "dimension_numbers": FakeGatherDimensionNumbers(
            offset_dims=(3, 4),
            collapsed_slice_dims=(2, 3),
            operand_batching_dims=(0, 1),
            start_indices_batching_dims=(0, 1),
            start_index_map=(2, 3)
        ),
        "slice_sizes": [1, 1, 1, 1, 4, 4],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": -99.0
    })

    return list_of_inputs

generated_inputs["jax.lax.gather_4"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_4'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_4'], lib="jax", suffix=4)
