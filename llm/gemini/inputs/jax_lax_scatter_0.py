
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.lax as lax
import numpy as np


class ScatterDimensionNumbers:

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
        self.scatter_indices_batching_dims = tuple(scatter_indices_batching_dims)

    def __len__(self):
        return 5

    def __iter__(self):
        return iter(
            (
                self.update_window_dims,
                self.inserted_window_dims,
                self.scatter_dims_to_operand_dims,
                self.operand_batching_dims,
                self.scatter_indices_batching_dims,
            )
        )

    def __getitem__(self, idx):
        return (
            self.update_window_dims,
            self.inserted_window_dims,
            self.scatter_dims_to_operand_dims,
            self.operand_batching_dims,
            self.scatter_indices_batching_dims,
        )[idx]

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __hash__(self):
        return hash(tuple(self))

    def __repr__(self):
        return (
            f"ScatterDimensionNumbers(update_window_dims={self.update_window_dims}, "
            f"inserted_window_dims={self.inserted_window_dims}, "
            f"scatter_dims_to_operand_dims={self.scatter_dims_to_operand_dims}, "
            f"operand_batching_dims={self.operand_batching_dims}, "
            f"scatter_indices_batching_dims={self.scatter_indices_batching_dims})"
        )

    def __array__(self, dtype=None, copy=None):
        return np.array([0], dtype=np.int32)


def scatter_inputs():
    list_of_inputs = []

    # Input 1: 1D operand, scattering 1D updates to specific indices
    operand = np.ones((5,), dtype=np.float32)
    scatter_indices = np.array([[1], [2], [4]], dtype=np.int32)
    updates = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    dimension_numbers = ScatterDimensionNumbers(
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

    # Input 2: 2D operand, scattering full row slices with clip mode
    operand = np.zeros((6, 4), dtype=np.float32)
    scatter_indices = np.array([[0], [3]], dtype=np.int32)
    updates = np.ones((2, 4), dtype=np.float32)
    dimension_numbers = ScatterDimensionNumbers(
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
        "unique_indices": True,
        "mode": "clip",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D operand, scattering scalar values to specific 2D coordinates
    operand = np.zeros((4, 5), dtype=np.int32)
    scatter_indices = np.array([[0, 1], [2, 3], [1, 4]], dtype=np.int32)
    updates = np.array([10, 20, 30], dtype=np.int32)
    dimension_numbers = ScatterDimensionNumbers(
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
        "unique_indices": False,
        "mode": "drop",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D operand with float64 dtype, scattering 2D matrix slices
    operand = np.ones((3, 4, 5), dtype=np.float64)
    scatter_indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.zeros((2, 4, 5), dtype=np.float64)
    dimension_numbers = ScatterDimensionNumbers(
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
        "mode": "fill",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D operand, scattering along column dimension
    operand = np.arange(15, dtype=np.float32).reshape(3, 5)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.ones((2, 3), dtype=np.float32) * -1.0
    dimension_numbers = ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "drop",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 operand with clip mode
    operand = np.arange(8, dtype=np.int64)
    scatter_indices = np.array([[1], [3], [2], [5]], dtype=np.int32)
    updates = np.array([100, 200, 300, 400], dtype=np.int64)
    dimension_numbers = ScatterDimensionNumbers(
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
        "mode": "clip",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D operand, updating 3D sub-volumes
    operand = np.ones((2, 3, 4, 2), dtype=np.float32)
    scatter_indices = np.array([[1]], dtype=np.int32)
    updates = np.full((1, 3, 4, 2), 5.0, dtype=np.float32)
    dimension_numbers = ScatterDimensionNumbers(
        update_window_dims=(1, 2, 3),
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

    # Input 8: Batch dimension in scatter_indices (shape: (2, 3, 1))
    operand = np.zeros((10, 5), dtype=np.float32)
    scatter_indices = np.array([[[1], [2], [3]], [[4], [5], [6]]], dtype=np.int32)
    updates = np.ones((2, 3, 5), dtype=np.float32)
    dimension_numbers = ScatterDimensionNumbers(
        update_window_dims=(2,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    input_dict = {
        "operand": operand,
        "scatter_indices": scatter_indices,
        "updates": updates,
        "dimension_numbers": dimension_numbers,
        "indices_are_sorted": False,
        "unique_indices": True,
        "mode": "drop",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element 1D operand update
    operand = np.array([5.0], dtype=np.float32)
    scatter_indices = np.array([[0]], dtype=np.int32)
    updates = np.array([10.0], dtype=np.float32)
    dimension_numbers = ScatterDimensionNumbers(
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

    # Input 10: 3D operand with int16 dtype, scattering 1D slice updates
    operand = np.zeros((4, 4, 4), dtype=np.int16)
    scatter_indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    updates = np.ones((3, 4), dtype=np.int16) * 7
    dimension_numbers = ScatterDimensionNumbers(
        update_window_dims=(1,),
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
        "mode": "clip",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.scatter"] = scatter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.scatter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.scatter'.")


check_valid('jax.lax.scatter', generated_inputs['jax.lax.scatter'], lib="jax", suffix=0)
