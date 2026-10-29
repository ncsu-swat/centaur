
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class GatherDimensionNumbers:
    _fields = (
        "offset_dims",
        "collapsed_slice_dims",
        "start_index_map",
        "operand_batching_dims",
        "start_indices_batching_dims",
    )

    def __init__(
        self,
        offset_dims=(),
        collapsed_slice_dims=(),
        start_index_map=(),
        operand_batching_dims=(),
        start_indices_batching_dims=(),
    ):
        self.offset_dims = tuple(offset_dims)
        self.collapsed_slice_dims = tuple(collapsed_slice_dims)
        self.start_index_map = tuple(start_index_map)
        self.operand_batching_dims = tuple(operand_batching_dims)
        self.start_indices_batching_dims = tuple(start_indices_batching_dims)
        self._flat = (
            self.offset_dims
            + self.collapsed_slice_dims
            + self.start_index_map
            + self.operand_batching_dims
            + self.start_indices_batching_dims
        )

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, idx):
        return self._flat[idx]

    def __iter__(self):
        return iter(self._flat)

    def __hash__(self):
        return hash((
            self.offset_dims,
            self.collapsed_slice_dims,
            self.start_index_map,
            self.operand_batching_dims,
            self.start_indices_batching_dims,
        ))

    def __eq__(self, other):
        if not isinstance(other, GatherDimensionNumbers):
            return False
        return (
            self.offset_dims == other.offset_dims
            and self.collapsed_slice_dims == other.collapsed_slice_dims
            and self.start_index_map == other.start_index_map
            and self.operand_batching_dims == other.operand_batching_dims
            and self.start_indices_batching_dims
            == other.start_indices_batching_dims
        )

    def __repr__(self):
        return (
            f"GatherDimensionNumbers(offset_dims={self.offset_dims},"
            f" collapsed_slice_dims={self.collapsed_slice_dims},"
            f" start_index_map={self.start_index_map},"
            f" operand_batching_dims={self.operand_batching_dims},"
            f" start_indices_batching_dims={self.start_indices_batching_dims})"
        )


def gather_inputs():
    list_of_inputs = []

    # Input 1: 1D operand, 1D gathering with collapsed slice
    operand = np.arange(10, dtype=np.float32)
    start_indices = np.array([[1], [3], [5], [7]], dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (1,),
                "unique_indices": True,
                "indices_are_sorted": True,
                "mode": "clip",
                "fill_value": 0,
            }
        )
    )

    # Input 2: 2D operand, gathering rows
    operand = np.random.randn(8, 5).astype(np.float32)
    start_indices = np.array([[0], [2], [4]], dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (1, 5),
                "unique_indices": True,
                "indices_are_sorted": True,
                "mode": "fill",
                "fill_value": -1,
            }
        )
    )

    # Input 3: 2D operand, gathering elements with 2D coordinates
    operand = np.arange(42, dtype=np.int32).reshape(6, 7)
    start_indices = np.array([[0, 1], [2, 3], [4, 5], [1, 2]], dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (1, 1),
                "unique_indices": False,
                "indices_are_sorted": False,
                "mode": "promise_in_bounds",
                "fill_value": 0,
            }
        )
    )

    # Input 4: 2D operand, slice gathering without collapsing
    operand = np.random.randn(12, 10).astype(np.float64)
    start_indices = np.array([[0], [5]], dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(1, 2), collapsed_slice_dims=(), start_index_map=(0,)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (3, 4),
                "unique_indices": False,
                "indices_are_sorted": True,
                "mode": "clip",
                "fill_value": 0,
            }
        )
    )

    # Input 5: 3D operand, gathering 2D slices along leading dimension
    operand = np.random.randn(5, 6, 7).astype(np.float32)
    start_indices = np.array([[1], [3], [0]], dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(1, 2), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (1, 6, 7),
                "unique_indices": True,
                "indices_are_sorted": False,
                "mode": "drop",
                "fill_value": 99,
            }
        )
    )

    # Input 6: 3D operand, multidimensional batch of start indices
    operand = np.arange(4 * 8 * 16, dtype=np.float32).reshape(4, 8, 16)
    start_indices = np.zeros((2, 3, 2), dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(2,), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (1, 1, 4),
                "unique_indices": False,
                "indices_are_sorted": False,
                "mode": "fill",
                "fill_value": 0,
            }
        )
    )

    # Input 7: 1D operand, multiple overlapping slices
    operand = np.linspace(0, 100, 20, dtype=np.float32)
    start_indices = np.array([[0], [2], [4], [6], [8]], dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (3,),
                "unique_indices": False,
                "indices_are_sorted": True,
                "mode": "clip",
                "fill_value": 0,
            }
        )
    )

    # Input 8: 2D operand, gathering columns
    operand = np.random.randn(4, 9).astype(np.float32)
    start_indices = np.array([[1], [3], [5]], dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(0,), collapsed_slice_dims=(1,), start_index_map=(1,)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (4, 1),
                "unique_indices": True,
                "indices_are_sorted": False,
                "mode": "promise_in_bounds",
                "fill_value": 0,
            }
        )
    )

    # Input 9: 4D operand, slice gathering
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    start_indices = np.array([[0], [1]], dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(1, 2, 3), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (1, 3, 4, 5),
                "unique_indices": True,
                "indices_are_sorted": True,
                "mode": "clip",
                "fill_value": 0,
            }
        )
    )

    # Input 10: 3D operand, point gathering across all 3 dimensions
    operand = np.arange(125, dtype=np.int64).reshape(5, 5, 5)
    start_indices = np.array(
        [[0, 0, 0], [1, 2, 3], [4, 4, 4], [2, 1, 0]], dtype=np.int32
    )
    dnums = GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0, 1, 2),
        start_index_map=(0, 1, 2),
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (1, 1, 1),
                "unique_indices": False,
                "indices_are_sorted": False,
                "mode": "fill",
                "fill_value": -999,
            }
        )
    )

    # Input 11: 1D boolean operand
    operand = np.array([True, False, True, True, False, False], dtype=np.bool_)
    start_indices = np.array([[0], [2], [4]], dtype=np.int32)
    dnums = GatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "dimension_numbers": dnums,
                "slice_sizes": (1,),
                "unique_indices": True,
                "indices_are_sorted": True,
                "mode": "clip",
                "fill_value": 1,
            }
        )
    )

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
