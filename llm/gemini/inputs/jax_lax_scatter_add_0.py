
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class ScatterDims:

    def __init__(
        self,
        update_window_dims,
        inserted_window_dims,
        scatter_dims_to_operand_dims,
        operand_batching_dims=(),
        scatter_indices_batching_dims=(),
    ):
        self.update_window_dims = tuple(update_window_dims)
        self.inserted_window_dims = tuple(inserted_window_dims)
        self.scatter_dims_to_operand_dims = tuple(scatter_dims_to_operand_dims)
        self.operand_batching_dims = tuple(operand_batching_dims)
        self.scatter_indices_batching_dims = tuple(
            scatter_indices_batching_dims
        )

    def __hash__(self):
        return hash(
            (
                self.update_window_dims,
                self.inserted_window_dims,
                self.scatter_dims_to_operand_dims,
                self.operand_batching_dims,
                self.scatter_indices_batching_dims,
            )
        )

    def __eq__(self, other):
        if isinstance(other, ScatterDims):
            return (
                self.update_window_dims == other.update_window_dims
                and self.inserted_window_dims == other.inserted_window_dims
                and self.scatter_dims_to_operand_dims
                == other.scatter_dims_to_operand_dims
                and self.operand_batching_dims == other.operand_batching_dims
                and self.scatter_indices_batching_dims
                == other.scatter_indices_batching_dims
            )
        return False

    def __len__(self):
        return 5

    def __iter__(self):
        return iter([0, 0, 0, 0, 0])

    def __getitem__(self, idx):
        return [
            self.update_window_dims,
            self.inserted_window_dims,
            self.scatter_dims_to_operand_dims,
            self.operand_batching_dims,
            self.scatter_indices_batching_dims,
        ][idx]

    def __array__(self, dtype=None, copy=None):
        return np.array([0, 0], dtype=dtype)


def scatter_add_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, point updates
    operand = np.zeros((10,), dtype=np.float32)
    scatter_indices = np.array([[1], [4], [7]], dtype=np.int32)
    updates = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "promise_in_bounds",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, slice updates along rows
    operand = np.ones((5, 4), dtype=np.float32)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.full((2, 4), 5.0, dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": False,
        "mode": "clip",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array, element updates
    operand = np.zeros((6, 6), dtype=np.int32)
    scatter_indices = np.array([[0, 1], [2, 3], [4, 5], [1, 2]], dtype=np.int32)
    updates = np.array([10, 20, 30, 40], dtype=np.int32)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "fill",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, 2D slice updates
    operand = np.random.randn(4, 4, 4).astype(np.float64)
    scatter_indices = np.array([[1], [3]], dtype=np.int64)
    updates = np.random.randn(2, 4, 4).astype(np.float64)
    dimension_numbers = ScatterDims(
        update_window_dims=(1, 2),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "drop",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32 array, non-unique indices with negative update values
    operand = np.full((8,), 2.0, dtype=np.float32)
    scatter_indices = np.array([[2], [2], [5]], dtype=np.int32)
    updates = np.array([-1.5, -2.5, 3.0], dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": False,
        "mode": "promise_in_bounds",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array, column updates
    operand = np.zeros((4, 5), dtype=np.float32)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.ones((2, 4), dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(1,),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "clip",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array with 2D batch of scatter indices
    operand = np.ones((8, 8, 8), dtype=np.float32)
    scatter_indices = np.array(
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[1, 2], [3, 4]]], dtype=np.int32
    )
    updates = np.random.randn(3, 2, 8).astype(np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(2,),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": False,
        "mode": "drop",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int64 array with sorted indices
    operand = np.arange(12, dtype=np.int64)
    scatter_indices = np.array([[0], [3], [6], [9]], dtype=np.int32)
    updates = np.array([100, 200, 300, 400], dtype=np.int64)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": True,
        "unique_indices": True,
        "mode": "fill",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, element updates
    operand = np.ones((4, 4), dtype=np.float32)
    scatter_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
    updates = np.array([1.5, -2.0, 3.5], dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "promise_in_bounds",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array with 2D window updates
    operand = np.zeros((3, 5, 4, 2), dtype=np.float32)
    scatter_indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    updates = np.ones((2, 4, 2), dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(1, 2),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": False,
        "mode": "drop",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.scatter_add"] = scatter_add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.scatter_add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.scatter_add'.")


check_valid('jax.lax.scatter_add', generated_inputs['jax.lax.scatter_add'], lib="jax", suffix=0)
