
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax.lax as lax
import numpy as np


class ScatterDims:

    def __init__(
        self,
        update_window_dims,
        inserted_window_dims,
        scatter_dims_to_operand_dims,
    ):
        self._dnums = lax.ScatterDimensionNumbers(
            update_window_dims=update_window_dims,
            inserted_window_dims=inserted_window_dims,
            scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
        )

    def __getattr__(self, name):
        return getattr(self._dnums, name)

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(self._dnums)

    def __getitem__(self, index):
        return self._dnums[index]

    def __hash__(self):
        return hash(self._dnums)

    def __eq__(self, other):
        if isinstance(other, ScatterDims):
            return self._dnums == other._dnums
        return self._dnums == other

    def __repr__(self):
        return repr(self._dnums)


def scatter_mul_inputs():
    list_of_inputs = []

    # Input 1: 1D array, basic scatter multiply
    operand = np.ones((5,), dtype=np.float32)
    scatter_indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": False,
            "unique_indices": True,
            "mode": "drop",
        }
    )

    # Input 2: 1D integer array with duplicate indices
    operand = np.array([1, 2, 3, 4], dtype=np.int32)
    scatter_indices = np.array([[1], [1]], dtype=np.int32)
    updates = np.array([2, 5], dtype=np.int32)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": True,
            "unique_indices": False,
            "mode": "clip",
        }
    )

    # Input 3: 2D array, scattering full rows
    operand = np.ones((4, 3), dtype=np.float32)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[2.0, 3.0, 4.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": True,
            "unique_indices": True,
            "mode": "fill",
        }
    )

    # Input 4: 2D array with float64, scattering individual elements with negative values
    operand = np.ones((3, 3), dtype=np.float64)
    scatter_indices = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    updates = np.array([-2.0, -3.0, 4.0], dtype=np.float64)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": False,
            "unique_indices": True,
            "mode": "promise_in_bounds",
        }
    )

    # Input 5: 3D array, scattering 2D slices
    operand = np.ones((2, 3, 4), dtype=np.float32)
    scatter_indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.full((2, 3, 4), 2.0, dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(1, 2),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": True,
            "unique_indices": True,
            "mode": "drop",
        }
    )

    # Input 6: 2D array with out-of-bounds indices and mode='clip'
    operand = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scatter_indices = np.array([[5, 5]], dtype=np.int32)
    updates = np.array([2.0], dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": True,
            "unique_indices": True,
            "mode": "clip",
        }
    )

    # Input 7: 1D int64 array with negative updates
    operand = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    scatter_indices = np.array([[0], [4]], dtype=np.int64)
    updates = np.array([2, -1], dtype=np.int64)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": True,
            "unique_indices": True,
            "mode": "drop",
        }
    )

    # Input 8: 3D array, scattering 1D slices
    operand = np.ones((3, 4, 5), dtype=np.float32)
    scatter_indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    updates = np.full((2, 5), 3.0, dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(1,),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": False,
            "unique_indices": True,
            "mode": "fill",
        }
    )

    # Input 9: 1D array with positive float32 updates
    operand = np.ones((6,), dtype=np.float32)
    scatter_indices = np.array([[1], [4]], dtype=np.int32)
    updates = np.array([0.5, 2.5], dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": True,
            "unique_indices": True,
            "mode": "drop",
        }
    )

    # Input 10: 4D array point updates
    operand = np.ones((2, 2, 2, 2), dtype=np.float32)
    scatter_indices = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    updates = np.array([0.5, 1.5], dtype=np.float32)
    dimension_numbers = ScatterDims(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2, 3),
        scatter_dims_to_operand_dims=(0, 1, 2, 3),
    )
    list_of_inputs.append(
        {
            "operand": operand,
            "scatter_indices": scatter_indices,
            "updates": updates,
            "dimension_numbers": dimension_numbers,
            "indices_are_sorted": True,
            "unique_indices": True,
            "mode": "promise_in_bounds",
        }
    )

    return list_of_inputs


generated_inputs["jax.lax.scatter_mul"] = scatter_mul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.scatter_mul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.scatter_mul'.")


check_valid('jax.lax.scatter_mul', generated_inputs['jax.lax.scatter_mul'], lib="jax", suffix=0)
