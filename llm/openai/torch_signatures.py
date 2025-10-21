signatures = {}
signatures["torch.DoubleStorage"] = {
    "args": {
        "size": "integer"  # Could also be a tuple of integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ShortStorage"] = {
    "args": {
        "size": "integer" # Could also be tuple, but documentation implies integer
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.abs"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.abs_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could be None as well
    },
    "inner": {},
}
signatures["torch.absolute"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {},
}
signatures["torch.acos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.acos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # potentially could be None, but tensor is more specific
    },
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.add"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Could also be "float" or "integer"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addbmm"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor"
    },
    "kwargs": {
        "beta": "float",  # could be integer
        "alpha": "float", # could be integer
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addcdiv"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",  # could be integer or float, defaulting to float
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addcmul"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",  # could be integer as well, depending on input type
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addmm"] = {
    "args": {
        "input": "tensor",
        "mat1": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype",  # could be None
        "beta": "float",
        "alpha": "float",
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.addmv"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "beta": "float",  # could be integer
        "alpha": "float", # could be integer
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addmv_"] = {
    "args": {
        "input": "tensor",
        "vec": "tensor",
        "mat": "tensor"
    },
    "kwargs": {
        "beta": "float", # Could also be a tensor
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addr"] = {
    "args": {
        "input": "tensor",
        "vec1": "tensor",
        "vec2": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.adjoint"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.alias_copy"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.all_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.all_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer" # or tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.allclose"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rtol": "float",
        "atol": "float",
        "equal_nan": "boolean"
    },
    "inner": {}
}
signatures["torch.amax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",  # could also be tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.amin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # could also be tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.aminmax"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",  # could be None
        "keepdim": "boolean",
        "out": "tuple"  # Expects a tuple of tensors
    },
    "inner": {}
}
signatures["torch.angle"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.any_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None
    },
    "inner": {}
}

signatures["torch.any_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # or tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.arange"] = {
    "args": {
        "start": "float",  # Could also be integer
        "end": "float",  # Could also be integer
        "step": "float"  # Could also be integer
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # Assuming layout is a string enum
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.arccos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.arccos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arccosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well
    },
    "inner": {},
}
signatures["torch.arcsin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {},
}
signatures["torch.arcsin_"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list
    },
    "kwargs": {
        "out": "tensor" # Could also be None
    },
    "inner": {}
}
signatures["torch.arcsinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {},
}
signatures["torch.arcsinh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.arctan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could be None as well, but tensor is more appropriate given the documentation
    },
    "inner": {},
}
signatures["torch.arctan_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.arctanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {},
}
signatures["torch.arctanh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.are_deterministic_algorithms_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.argmax_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.argmax_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean" # could also be None
    },
    "inner": {}
}
signatures["torch.argmin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # could be None
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.argsort"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",  # could also be None
        "descending": "boolean"
    },
    "kwargs": {
        "stable": "boolean"
    },
    "inner": {}
}
signatures["torch.argwhere"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.as_strided"] = {
    "args": {
        "input": "tensor",
        "size": "tuple",
        "stride": "tuple",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.as_strided_copy"] = {
    "args": {
        "input": "tensor",
        "size": "tuple",
        "stride": "tuple",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.as_tensor_1"] = {
    "args": {
        "data": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string"  # device is skipped
    },
    "inner": {},
}
signatures["torch.as_tensor_2"] = {
    "args": {
        "data": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string"  # device is skipped
    },
    "inner": {},
}
signatures["torch.as_tensor_3"] = {
    "args": {
        "data": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string"  # device is skipped
    },
    "inner": {},
}
signatures["torch.asarray"] = {
    "args": {
        "obj": "tensor"  # Could be a tensor, NumPy array, DLPack capsule, object with buffer protocol, scalar, or sequence of scalars
    },
    "kwargs": {
        "dtype": "dtype",  # The datatype of the returned tensor
        "copy": "boolean",  # Controls whether the returned tensor shares memory with obj
        "requires_grad": "boolean"  # Whether the returned tensor requires grad
    },
    "inner": {}
}
signatures["torch.asin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well, but tensor is more likely
    },
    "inner": {},
}
signatures["torch.asin_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor" # could be None as well
    },
    "inner": {}
}
signatures["torch.asinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.asinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well, but tensor is more specific
    },
    "inner": {},
}
signatures["torch.atan2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # potentially None, but tensor is the most specific
    },
    "inner": {}
}
signatures["torch.atan_"] = {
    "args": {
        "tensor": "tensor",
        "other": "tensor"  # Could be a number, so potentially float as well
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.atanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well, but tensor is more common
    },
    "inner": {},
}
signatures["torch.atanh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.atleast_1d"] = {
    "args": {
        "input": "tensor"  # could also be tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_2d"] = {
    "args": {
        "input": "tensor"  # could also be tensor_list
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.atleast_3d"] = {
    "args": {
        "input": "tensor"  # Could be tensor_list as well, but documentation says tensor.
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.autocast"] = {
    "args": {
        "enabled": "boolean" # can also be a callable that returns a boolean
    },
    "kwargs": {
        "cache": "boolean",
        "dtype": "dtype" # or torch.dtype
    },
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.autocast_decrement_nesting"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.autocast_increment_nesting"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.baddbmm"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype",  # could be None
        "beta": "float",
        "alpha": "float",
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.bartlett_window"] = {
    "args": {
        "length": "integer",
        "periodic": "boolean",
        "alpha": "float" # Could also be integer, assuming float for now
    },
    "kwargs": {
        "dtype": "dtype", # Could be torch.dtype
        "layout": "string" # Could be torch.layout
    },
    "inner": {},
}
signatures["torch.bilinear"] = {
    "args": {
        "input": "tensor",
        "x1": "float",
        "x2": "float",
        "y1": "float",
        "y2": "float",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bincount"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "weights": "tensor",  # Could be None, but defaulting to tensor
        "minlength": "integer"
    },
    "inner": {},
}
signatures["torch.bitwise_and"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bitwise_left_shift"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bitwise_not"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could be None too, but tensor seems most common
    },
    "inner": {},
}
signatures["torch.bitwise_or"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Should it be tensor or optional tensor?
    },
    "inner": {}
}
signatures["torch.bitwise_right_shift"] = {
    "args": {
        "input": "tensor",
        "shift": "integer" # Could also be tensor, but integer seems more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bitwise_xor"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could be None as well
    },
    "inner": {},
}
signatures["torch.blackman_window"] = {
    "args": {
        "length": "integer",  # could also be tensor
        "periodic": "boolean"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string" # could also be a tuple
    },
    "inner": {},
}
signatures["torch.block_diag"] = {
    "args": {
        "matrices": "tensor_list"  # could also be a single tensor
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bmm"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype",  # could be None
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.broadcast_shapes"] = {
    "args": {
        "shapes": "tensor_list"  # Could also be a list of integers
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.broadcast_tensors"] = {
    "args": {
        "tensors": "tensor_list"  # could also be a tuple of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.broadcast_to"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple"  # Could also be a list of integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bucketize"] = {
    "args": {
        "input": "tensor",
        "boundaries": "tensor"
    },
    "kwargs": {
        "out_int32": "boolean", # could also be dtype
        "right": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.cartesian_prod"] = {
    "args": {
        "inputs": "tensor_list"  # could also be a tuple of tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cat"] = {
    "args": {
        "tensors": "tensor_list",
        "dim": "integer"  # Could also be a tuple, but the documentation doesn't explicitly state it
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.cdist"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "p": "float",  # could also be integer
        "compute_mode": "string"
    },
    "inner": {}
}
signatures["torch.ceil"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well, but tensor is more appropriate.
    },
    "inner": {}
}
signatures["torch.ceil_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could be None as well
    },
    "inner": {}
}
signatures["torch.celu"] = {
    "args": {
        "input": "tensor",
        "alpha": "float"  # Could be tensor as well, but float is more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.chain_matmul"] = {
    "args": {
        "tensors": "tensor_list" # Could also be a tuple of tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.channel_shuffle"] = {
    "args": {
        "input": "tensor",
        "groups": "integer"  # could also be a tuple, but integer seems more common
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cholesky"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.cholesky_inverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cholesky_solve"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "upper": "boolean"  # Could be boolean or None, defaulting to True.
    },
    "inner": {}
}
signatures["torch.chunk"] = {
    "args": {
        "input": "tensor",
        "chunks": "integer",
        "dim": "integer"  # could also be a default value
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.clamp"] = {
    "args": {
        "input": "tensor",
        "min": "float",  # could also be tensor
        "max": "float"  # could also be tensor
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.clamp_max"] = {
    "args": {
        "input": "tensor",
        "max": "float"  # or tensor, but float seems more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.clear_autocast_cache"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.clip"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "tensor",  # could also be float or integer
        "max": "tensor",  # could also be float or integer
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clip_"] = {
    "args": {
        "input": "tensor",
        "min": "float",
        "max": "float"
    },
    "kwargs": {
        "out": "tensor" #Could be None as well
    },
    "inner": {},
}
signatures["torch.column_stack"] = {
    "args": {
        "tensors": "tensor_list"  # could also be a tuple of tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.combinations"] = {
    "args": {
        "input": "tensor",
        "r": "integer"
    },
    "kwargs": {
        "with_replacement": "boolean"
    },
    "inner": {},
}
signatures["torch.complex"] = {
    "args": {
        "real": "tensor",  # could be half, float or double, but all are tensors
        "imag": "tensor"  # must be the same dtype as real
    },
    "kwargs": {
        "out": "tensor"  # depends on the dtype of input tensors
    },
    "inner": {}
}
signatures["torch.concat"] = {
    "args": {
        "tensors": "tensor_list",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {},
}
signatures["torch.concatenate"] = {
    "args": {
        "tensors": "tensor_list",
    },
    "kwargs": {
        "axis": "integer",  # Could also be "int"
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.conj"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.conj_physical"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.conj_physical_"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.copysign"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # or float, but tensor is more general
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.corrcoef"] = {
    "args": {
        "input": "tensor"  # Could also be a scalar or 1D vector, but "tensor" covers those cases.
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.cos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.cosh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.count_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer", # Could be a tuple of integers as well
        "keepdim": "boolean",
        "dtype": "dtype" # might be None
    },
    "inner": {},
}
signatures["torch.cross"] = {
    "args": {
        "input": "tensor",
        "other": "tensor",
        "dim": "integer" # Could also be None
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.crow_indices_copy"] = {
    "args": {
        "graph": "tensor",
        "indices": "tensor",
        "out": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cudnn_affine_grid_generator"] = {
    "args": {
        "batch_size": "integer",
        "num_features": "integer",
        "height": "integer",
        "width": "integer"
    },
    "kwargs": {
        "dtype": "dtype" # Should it be torch.dtype?
    },
    "inner": {}
}
signatures["torch.cummax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tuple"  # Should it be tensor? or tuple of tensors?
    },
    "inner": {}
}
signatures["torch.cummin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tuple"  # could be None, but tuple seems more accurate based on documentation
    },
    "inner": {}
}
signatures["torch.cumprod"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.cumsum"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.deg2rad"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.dequantize_1"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.dequantize_2"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.det"] = {
    "args": {
        "input": "tensor"  # Should it be tensor_list as well? No, documentation says tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.device"] = {
    "args": {
        "type": "string", # Could also be 'integer', but string seems more representative of device names.
        "index": "integer" # Could be tensor as well, but integer seems to be the common use case.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.diag"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.diag_embed"] = {
    "args": {
        "input": "tensor",
        "offset": "integer" # could also be tensor, but integer is more common
    },
    "kwargs": {
        "dim1": "integer",
        "dim2": "integer"
    },
    "inner": {},
}
signatures["torch.diagflat"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "offset": "integer"  # Could be int, but integer is more general
    },
    "inner": {}
}
signatures["torch.diagonal"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "offset": "integer",
        "dim1": "integer",
        "dim2": "integer",
    },
    "inner": {},
}
signatures["torch.diff"] = {
    "args": {
        "input": "tensor",
        "n": "integer",
        "dim": "integer",
        "prepend": "tensor",  # Could also be None
        "append": "tensor"  # Could also be None
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.digamma"] = {
    "args": {
        "input": "tensor"  # Could be tensor_list as well, but documentation doesn't mention that
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.dist"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "p": "float"  # Could be integer as well, but float seems more general
    },
    "inner": {},
}
signatures["torch.div"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # or float/integer
    },
    "kwargs": {
        "rounding_mode": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.divide"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Could be a number (float or integer)
    },
    "kwargs": {
        "rounding_mode": "string",  # Could be None
        "out": "tensor"  # Could be None
    },
    "inner": {},
}
signatures["torch.dot"] = {
    "args": {
        "input": "tensor",
        "tensor": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.dsplit"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"  # could also be integer or tuple
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.dstack"] = {
    "args": {
        "tensors": "tensor_list"  # could also be a tuple of tensors
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.einsum"] = {
    "args": {
        "equation": "string",
        "operands": "tensor_list"  # Could be a single tensor as well
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.empty"] = {
    "args": {
        "size": "list",  # Could also be a tuple or variable number of integers
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "string",
    },
    "inner": {},
}
signatures["torch.empty_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype", # Could be torch.dtype instead
        "layout": "string", # Could be torch.layout instead
        "device": "string", #device not included
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.empty_strided"] = {
    "args": {
        "size": "tuple",  # or list
        "stride": "tuple",  # or list
        "dtype": "dtype",
        "layout": "string",
        "device": "string", # Skipped based on instructions
        "requires_grad": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.eq"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # could also be float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.erf"] = {
    "args": {
        "input": "tensor"  # Could be tensor or tensor_list
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.erf_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.erfc"] = {
    "args": {
        "input": "tensor"  # Could be tensor or tensor_list
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.erfc_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.exp"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.exp2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.expand_copy"] = {
    "args": {
        "input": "tensor",
        "sizes": "tuple" # or list, but tuple seems more specific
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.expm1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.eye"] = {
    "args": {
        "n": "integer",
        "m": "integer"  # Could be optional, but documentation states it defaults to n
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # torch.layout is a string
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.fake_quantize_per_channel_affine"] = {
    "args": {
        "input": "tensor",
        "fake_quant_params": "tuple" # Could also be a dict, but tuple is more common for per-channel
    },
    "kwargs": {
        "quantize_dtype": "dtype",
        "scale": "tensor",
        "zero_point": "tensor",
        "observer": "object" # Could be a more specific type if known
    },
    "inner": {}
}
signatures["torch.fake_quantize_per_tensor_affine"] = {
    "args": {
        "input": "tensor",
        "quant_min": "float",
        "quant_max": "float",
        "scale": "float",
        "zero_point": "integer"
    },
    "kwargs": {
        "fake_quant_mode": "string", # Could be enum?
        "dtype": "dtype" # Could be torch.dtype?
    },
    "inner": {}
}
signatures["torch.fft.fft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",  # could be None
        "dim": "integer"
    },
    "kwargs": {
        "norm": "string",
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.fft.fft2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "s": "tuple",  # Could be a list as well
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.fftfreq"] = {
    "args": {
        "n": "integer",
        "d": "float"  # Could be interpreted as float, it's a scaling factor.
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # Assuming layout is a string representing the layout type
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.fft.fftn"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "s": "tuple",  # could also be None
        "dim": "tuple",  # could also be None
        "norm": "string",  # could also be None
        "out": "tensor"  # could also be None
    },
    "inner": {},
}
signatures["torch.fft.fftshift"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"  # Could also be tuple[int]
    },
    "inner": {},
}
signatures["torch.fft.hfft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",  # could also be None
        "dim": "integer"
    },
    "kwargs": {
        "norm": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.ifft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",  # could also be None
        "dim": "integer"
    },
    "kwargs": {
        "norm": "string",
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.fft.ifft2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "s": "tuple",  # Could be a list as well
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.ifftn"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "s": "tuple",  # Could be a list as well
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.ifftshift"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple"  # Could also be integer, creating a new signature might be necessary
    },
    "inner": {}
}
signatures["torch.fft.ihfft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",  # could also be None
        "dim": "integer"
    },
    "kwargs": {
        "norm": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.irfft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",  # could also be None
        "dim": "integer"
    },
    "kwargs": {
        "norm": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.irfft2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "s": "tuple",  # can be None
        "dim": "tuple",  # can be None, default is last two dims
        "norm": "string",  # can be None, default is "backward"
        "out": "tensor"  # can be None
    },
    "inner": {},
}
signatures["torch.fft.irfftn"] = {
    "args": {
        "input": "tensor",
        "s": "tuple",
        "dim": "tuple"
    },
    "kwargs": {
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.rfft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",  # could be None
        "dim": "integer"
    },
    "kwargs": {
        "norm": "string",
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.fft.rfft2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "s": "tuple",  # Could be a list as well
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.rfftfreq"] = {
    "args": {
        "n": "integer",
        "d": "float"  # could also be integer, but float is more general
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # torch.layout is a string
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.fft.rfftn"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "s": "tuple",  # could also be None
        "dim": "tuple",  # could also be None
        "norm": "string",  # could also be None
        "out": "tensor"  # could also be None
    },
    "inner": {},
}
signatures["torch.finfo"] = {
    "args": {
        "dtype": "dtype"  # Could be a string representing the dtype as well
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.fix"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None, but tensor is more specific
    },
    "inner": {},
}
signatures["torch.flatten"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "start_dim": "integer",  # could also be None
        "end_dim": "integer",  # could also be None
    },
    "inner": {},
}
signatures["torch.flip"] = {
    "args": {
        "input": "tensor",
        "dims": "list"  # could also be tuple
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fliplr"] = {
    "args": {
        "input": "tensor"  # should be tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.flipud"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.float_power"] = {
    "args": {
        "input": "tensor",
        "exponent": "float" # Could also be tensor, but float seems more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.floor_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.floor_divide"] = {
    "args": {
        "input": "tensor",
        "other": "tensor" # or integer, but tensor seems more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fmax"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.fmin"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.frac"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.frexp"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tuple"  # Should it be tensor instead of tuple?
    },
    "inner": {},
}
signatures["torch.from_dlpack"] = {
    "args": {
        "data": "tensor",
        "dtype": "dtype" # could also be string for the dtype name
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.from_numpy"] = {
    "args": {
        "input": "tensor" # Could also be list, but tensor seems most accurate
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.full"] = {
    "args": {
        "size": "list",  # or tuple, or torch.Size
        "fill_value": "float"  # Scalar can be int or float, assuming float as default
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # torch.layout
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.full_like"] = {
    "args": {
        "input": "tensor",
        "size": "tuple"  # Could also be integer, but tuple seems more common.
    },
    "kwargs": {
        "fill_value": "float",
        "dtype": "dtype",
        "layout": "string",
        "device": "string", # Skipping device
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.gather"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor"  # Could also be LongTensor, but tensor is more general
    },
    "kwargs": {
        "sparse_grad": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.gcd"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ge"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # or float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.get_autocast_cpu_dtype"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_autocast_xla_dtype"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.get_default_device"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_default_dtype"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_deterministic_debug_mode"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_device"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.get_file_path"] = {
    "args": {
        "url": "string" # Could also be a Path object, but string seems most general
    },
    "kwargs": {
        "cache_dir": "string",
        "force": "boolean"
    },
    "inner": {},
}
signatures["torch.get_num_threads"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_rng_state"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.gradient"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "float",  # could also be list of floats or list of tensors
        "dim": "integer",  # could also be list of integers
        "edge_order": "integer"
    },
    "inner": {}
}
signatures["torch.greater"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {},
}
signatures["torch.greater_equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could be None as well
    },
    "inner": {},
}
signatures["torch.gt"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # or float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.hamming_window"] = {
    "args": {
        "length": "integer",  # could also be a tuple
        "periodic": "boolean",
        "alpha": "float",
        "beta": "float",
        "window_type": "string" # although it accepts a constant, it's still a string
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string",  # Skip device
        "layout": "string"
    },
    "inner": {},
}
signatures["torch.hann_window"] = {
    "args": {
        "length": "integer",  # could also be tensor
        "periodic": "boolean"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # could be torch.layout
        "device": "string", # intentionally skipped
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.heaviside"] = {
    "args": {
        "input": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.histc"] = {
    "args": {
        "input": "tensor",
        "bins": "integer",
        "min": "float",
        "max": "float"
    },
    "kwargs": {
        "out": "tensor" # could be None as well
    },
    "inner": {}
}
signatures["torch.histogram"] = {
    "args": {
        "input": "tensor",
        "bins": "integer"  # or "tensor", creating separate signatures for each would be too complex
    },
    "kwargs": {
        "range": "tuple",
        "weight": "tensor",
        "density": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.hsplit"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"  # Can also be an integer
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.hspmm"] = {
    "args": {
        "mat1": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more appropriate
    },
    "inner": {},
}
signatures["torch.hstack"] = {
    "args": {
        "tensors": "tensor_list"  # could also be a single tensor
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.hypot"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.i0"] = {
    "args": {
        "input": "tensor"  # Could be tensor_list as well, but documentation doesn't explicitly state it.
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.igammac"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor" # could be None as well, but tensor is a good default
    },
    "inner": {},
}
signatures["torch.iinfo"] = {
    "args": {
        "torch_dtype": "dtype" # Could also be string representing the dtype
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.imag"] = {
    "args": {
        "input": "tensor"  # Expecting a tensor as input
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.index_add_1"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "src": "tensor"
    },
    "kwargs": {
        "reduction": "string" # Could be 'add', 'sum', 'mean', etc.
    },
    "inner": {}
}
signatures["torch.index_add_2"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "src": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.index_copy"] = {
    "args": {
        "input": "tensor",
        "index": "tensor",
        "source": "tensor"
    },
    "kwargs": {
        "index_type": "dtype"  # Could be integer or long, dtype is a good fit.
    },
    "inner": {},
}
signatures["torch.index_put"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "values": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.index_select_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.inner"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.inverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.is_anomaly_check_nan_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_anomaly_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_autocast_cache_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_autocast_cpu_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_autocast_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_autocast_ipu_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_autocast_xla_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_deterministic_algorithms_warn_only_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_floating_point"] = {
    "args": {
        "input": "tensor"  # Could be tensor or tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_grad_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_inference"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_inference_mode_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",  # Could also be None
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.is_same_size"] = {
    "args": {
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_storage"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_tensor"] = {
    "args": {
        "input": "tensor"  # Could also be 'object' as it accepts anything
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_warn_always_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.isclose"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rtol": "float",
        "atol": "float",
        "equal_nan": "boolean"
    },
    "inner": {}
}
signatures["torch.isfinite"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.isin"] = {
    "args": {
        "elements": "tensor",
        "test_elements": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",  # Could also be integer, but boolean seems more appropriate
        "invert": "boolean"
    },
    "inner": {},
}
signatures["torch.isinf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.isnan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.isneginf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.isposinf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.isreal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.CompilationUnit"] = {
    "args": {
        "fn": "tensor", # Could also be a callable
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # Shape: (*, ...)
        },
        "kwargs": {}
    }
}
signatures["torch.jit.CompilationUnit_1"] = {
    "args": {
        "fn": "list" # Could also be a callable
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor_list" # Shape: (*, ...)
        },
        "kwargs": {}
    }
}
signatures["torch.jit.CompilationUnit_2"] = {
    "args": {
        "fn": "tuple" # Could also be a callable
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tuple" # Shape: (*, ...)
        },
        "kwargs": {}
    }
}
signatures["torch.jit.Error"] = {
    "args": {
        "message": "string" # Could also be tensor, but string seems more appropriate based on documentation
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.ScriptWarning"] = {
    "args": {
        "message": "string",
        "node": "object" # Could be anything related to the graph, potentially a tensor or object
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.enable_onednn_fusion"] = {
    "args": {},
    "kwargs": {
        "enabled": "boolean"  # Could also be integer (0 or 1)
    },
    "inner": {},
}
signatures["torch.jit.ignore"] = {
    "args": {},
    "kwargs": {
        "drop": "boolean" # could also be None, but boolean seems more appropriate
    },
    "inner": {}
}
signatures["torch.jit.is_scripting"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.is_tracing"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.isinstance"] = {
    "args": {
        "obj": "tensor", # Could be Any, but assuming tensor for now
        "target_type": "list" # Could also be tuple, or basic types like int/bool
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.optimized_execution"] = {
    "args": {
        "module": "tensor"  # Could also be a callable, but defaulting to tensor for simplicity.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.script_if_tracing"] = {
    "args": {
        "module": "tensor" # Could be callable or Module. Assuming tensor for now.
    },
    "kwargs": {},
    "inner": {
        "args": {
            "args": "tuple", # Could be a tuple of tensors, or other types
            "kwargs": "dict" # Could be a dict of tensors, or other types
        },
        "kwargs": {}
    }
}
signatures["torch.jit.set_fusion_strategy"] = {
    "args": {
        "module": "tensor" # Could also be a Module object, treating it as a tensor for simplicity
    },
    "kwargs": {
        "strategy": "string"
    },
    "inner": {},
}
signatures["torch.jit.set_module"] = {
    "args": {
        "name": "string",
        "module": "tensor"  # Could also be a callable, but defaulting to tensor for now.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.strict_fusion"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.wait"] = {
    "args": {
        "future": "tensor" # Should be torch.jit.Future[T], but approximating with tensor
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.kaiser_window"] = {
    "args": {
        "window_length": "integer",
        "periodic": "boolean",
        "beta": "float",
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string" # Skipping device as requested.
    },
    "inner": {},
}
signatures["torch.kron"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.kthvalue"] = {
    "args": {
        "input": "tensor",
        "k": "integer",
        "dim": "integer"  # could be None
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"  # could be None
    },
    "inner": {}
}
signatures["torch.lcm"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ldexp"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Could be integer as well, but tensor is more general
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.ldexp_"] = {
    "args": {
        "input": "tensor",
        "exponent": "tensor" # Could also be integer
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.le"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Could also be scalar (float or integer)
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.lerp"] = {
    "args": {
        "input": "tensor",
        "end": "tensor",
        "weight": "float"  # Could also be a tensor
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.less"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {},
}
signatures["torch.less_equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.lgamma"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.linalg.cholesky"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.cholesky_ex"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "upper": "boolean", # Could also be None
        "lower": "boolean"  # Could also be None
    },
    "inner": {}
}
signatures["torch.linalg.det"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor" # could be None, but defaulting to tensor
    },
    "inner": {}
}
signatures["torch.linalg.eig"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.linalg.eigh"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "UPLO": "string",
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.linalg.eigvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor" # could be None, but represented as tensor for simplicity
    },
    "inner": {}
}
signatures["torch.linalg.eigvalsh"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "UPLO": "string",  # could also be 'L' or 'U'
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.linalg.householder_product"] = {
    "args": {
        "A": "tensor",
        "householders": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.linalg.inv"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could be None, but defaulting to tensor
    },
    "inner": {}
}
signatures["torch.linalg.lstsq"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "rcond": "float",  # could be None
        "driver": "string"  # could be None
    },
    "inner": {}
}
signatures["torch.linalg.lu"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "pivot": "boolean",
        "out": "tuple"  # Potentially a tuple of tensors, but documented as tuple
    },
    "inner": {}
}
signatures["torch.linalg.matrix_norm"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ord": "string",  # could also be integer or 'fro'
        "dim": "tuple",
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.linalg.matrix_power"] = {
    "args": {
        "input": "tensor",
        "n": "integer"  # Could be float, but integer seems more appropriate
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.matrix_rank"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "tol": "float",  # Could be integer as well, but float is more general
        "abs_tol": "float", # Could be integer as well, but float is more general
        "hermitian": "boolean",
    },
    "inner": {},
}
signatures["torch.linalg.multi_dot"] = {
    "args": {
        "tensors": "tensor_list"  # could also be a tuple of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.linalg.norm"] = {
    "args": {
        "A": "tensor",
    },
    "kwargs": {
        "ord": "float",  # Can be int, float, inf, -inf, 'fro', 'nuc', optional
        "dim": "tuple",  # Can be int, Tuple[int], optional
        "keepdim": "boolean",
        "out": "tensor",  # optional
        "dtype": "dtype",  # optional
    },
    "inner": {},
}
signatures["torch.linalg.pinv"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "atol": "float",  # or tensor
        "rtol": "float",  # or tensor
        "hermitian": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.linalg.qr"] = {
    "args": {
        "A": "tensor",
    },
    "kwargs": {
        "mode": "string",  # Could be reduced, complete, or r
        "out": "tuple"  # tuple of two tensors, but ignored if None
    },
    "inner": {},
}
signatures["torch.linalg.slogdet"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tuple" # Could be None, but tuple is more specific
    },
    "inner": {},
}
signatures["torch.linalg.solve"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "left": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.linalg.solve_ex"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "rcond": "float", # Could also be a tuple, but float is simpler
        "hermitian": "boolean"
    },
    "inner": {},
}
signatures["torch.linalg.solve_triangular"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "upper": "boolean", # could also be None
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.linalg.svd"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "full_matrices": "boolean",
        "driver": "string",
        "out": "tuple"  # Should be a tuple of tensors, but "tuple" is the closest match
    },
    "inner": {}
}
signatures["torch.linalg.svdvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "driver": "string",  # could be None
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.linalg.tensorinv"] = {
    "args": {
        "A": "tensor",
        "ind": "integer"
    },
    "kwargs": {
        "out": "tensor"  # Should it be tensor or None? Assuming tensor
    },
    "inner": {}
}
signatures["torch.linalg.tensorsolve"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "dims": "tuple",
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.linalg.vecdot"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "dim": "integer",  # could also be None
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.linalg.vector_norm"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "ord": "float", # Could also be integer or string ("fro")
        "dim": "integer",
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.linspace"] = {
    "args": {
        "start": "float",  # or tensor
        "end": "float",  # or tensor
        "steps": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",
        "device": "string", # Skipping device
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.log"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.log10"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well, but tensor is more general
    },
    "inner": {},
}
signatures["torch.log1p"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.log2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well, but tensor is more general
    },
    "inner": {},
}
signatures["torch.log_"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logaddexp"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.logaddexp2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.logcumsumexp"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.logdet"] = {
    "args": {
        "input": "tensor"  # Should be tensor of size (*, n, n)
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logical_and"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could be None as well
    },
    "inner": {},
}
signatures["torch.logical_not"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.logical_or"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Should it be tensor or None? Assuming tensor.
    },
    "inner": {}
}
signatures["torch.logical_xor"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Should it be tensor or None? Assuming tensor.
    },
    "inner": {}
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "eps": "float"  # Could be tensor as well, but float is more general
    },
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.logspace"] = {
    "args": {
        "start": "float",  # could also be tensor
        "end": "float",  # could also be tensor
        "steps": "integer"
    },
    "kwargs": {
        "base": "float",
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.logsumexp"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",  # or tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lt"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Could also be float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lu_solve"] = {
    "args": {
        "A": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.lu_unpack"] = {
    "args": {
        "lu": "tensor",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.manual_seed"] = {
    "args": {
        "generator": "tensor"  # Could also be integer, but tensor seems more common.
    },
    "kwargs": {
        "seed": "integer"
    },
    "inner": {}
}
signatures["torch.masked_scatter"] = {
    "args": {
        "input": "tensor",
        "mask": "tensor",
        "source": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.masked_select"] = {
    "args": {
        "input": "tensor",
        "mask": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.matmul"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.matrix_exp"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.matrix_power"] = {
    "args": {
        "input": "tensor",
        "n": "integer"  # Could be float, but integer seems more common for power
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.max_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.max_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"  # Should be a tuple of tensors, but treating as tensor for simplicity
    },
    "inner": {},
}

signatures["torch.max_3"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.maximum"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.mean_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.mean_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # or tuple
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.median_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.median_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer" # could also be None
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple" # (Tensor, Tensor)
    },
    "inner": {}
}
signatures["torch.meshgrid"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "indexing": "string"  # Could also be None
    },
    "inner": {}
}
signatures["torch.min_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.min_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"  # Could be tensor, tuple
    },
    "inner": {}
}

signatures["torch.min_3"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.minimum"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.miopen_batch_norm"] = {
    "args": {
        "input": "tensor",
        "running_mean": "tensor",
        "running_var": "tensor",
        "weight": "tensor",
        "bias": "tensor",
        "momentum": "float",
        "eps": "float"
    },
    "kwargs": {
        "training": "boolean",  # Could be boolean or None
        "track_running_stats": "boolean"
    },
    "inner": {},
}
signatures["torch.mm"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype",  # could be None
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.moveaxis"] = {
    "args": {
        "input": "tensor",
        "source": "integer", # can also be tuple
        "destination": "integer" # can also be tuple
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.movedim"] = {
    "args": {
        "input": "tensor",
        "source": "tuple", # Can also be integer
        "destination": "tuple" # Can also be integer
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.msort"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well
    },
    "inner": {}
}
signatures["torch.mul"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Could also be "float" or "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.multiply"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well
    },
    "inner": {},
}
signatures["torch.mv"] = {
    "args": {
        "input": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.nan_to_num"] = {
    "args": {
        "tensor": "tensor",
        "nan": "float",
        "posinf": "float",
        "neginf": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nanmean"] = {
    "args": {
        "input": "tensor",
        "dim": "integer", # or tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nanmedian_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.nanmedian_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # Could also be None
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple" # Expects a tuple of tensors (values, indices)
    },
    "inner": {}
}
signatures["torch.nansum_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nansum_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # or tuple of ints
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.narrow"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "start": "integer", # can also be tensor
        "length": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.narrow_copy"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "start": "integer",
        "length": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.native_channel_shuffle"] = {
    "args": {
        "input": "tensor",
        "group": "integer" # Could also be a tuple, but prioritizing integer as the single signature
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.native_dropout"] = {
    "args": {
        "input": "tensor",
        "p": "float",
        "mode": "string",
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ne"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # or float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.neg"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.negative"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.negative_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float" # Could be a tensor as well, but float is more common.
    },
    "inner": {},
}
signatures["torch.nextafter"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.nn.AdaptiveAvgPool1d"] = {
    "args": {
        "output_size": "Union[int, tuple[int]]" # Could be int or tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveAvgPool2d"] = {
    "args": {
        "output_size": "tuple"  # Can be int or tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveAvgPool3d"] = {
    "args": {
        "output_size": "tuple"  # Could be int or tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveLogSoftmaxWithLoss"] = {
    "args": {
        "in_features": "integer",
        "n_classes": "integer",
        "cutoffs": "list"  # could also be tuple
    },
    "kwargs": {
        "div_value": "float",
        "head_bias": "boolean"
    },
    "inner": {
        "args": {
            "input_": "tensor",
            "target_": "tensor" # could be also integer
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveLogSoftmaxWithLoss_log_prob"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.AdaptiveLogSoftmaxWithLoss_predict"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.AdaptiveLogSoftmaxWithLoss_reset_parameters"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.AdaptiveMaxPool1d"] = {
    "args": {
        "output_size": "union[integer, tuple]" # could also be a list, but tuple is more specific
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveMaxPool2d"] = {
    "args": {
        "output_size": "tuple"  # Could be int or tuple
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveMaxPool3d"] = {
    "args": {
        "output_size": "tuple"  # Could be int or tuple
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AlphaDropout"] = {
    "args": {
        "p": "float"  # Should it be a tensor instead of float?
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool1d"] = {
    "args": {
        "kernel_size": "integer", # or tuple[int]
        "stride": "integer", # or tuple[int]
        "padding": "integer", # or tuple[int]
    },
    "kwargs": {
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool2d"] = {
    "args": {
        "kernel_size": "tuple",  # Can also be an integer
        "stride": "tuple",  # Can also be an integer, defaults to kernel_size
        "padding": "tuple",  # Can also be an integer, defaults to 0
    },
    "kwargs": {
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer",
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {},
    },
}
signatures["torch.nn.AvgPool3d"] = {
    "args": {
        "kernel_size": "tuple",  # could also be integer
        "stride": "tuple",  # could also be integer, defaults to kernel_size
        "padding": "tuple",  # could also be integer, defaults to 0
    },
    "kwargs": {
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer",
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {},
    }
}
signatures["torch.nn.BCELoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",  # Should it be tensor_list? Documentation mentions nbatch size
        "size_average": "boolean", # Deprecated
        "reduce": "boolean", # Deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.BCEWithLogitsLoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string",
        "pos_weight": "tensor"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.BatchNorm1d"] = {
    "args": {
        "num_features": "integer",
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # could be None
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.BatchNorm2d"] = {
    "args": {
        "num_features": "integer",
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # could be None
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype" # Added dtype
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.BatchNorm3d"] = {
    "args": {
        "num_features": "integer",
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Could also be None for cumulative moving average
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.Bilinear"] = {
    "args": {
        "in1_features": "integer",
        "in2_features": "integer",
        "out_features": "integer"
    },
    "kwargs": {
        "bias": "boolean"  # could also be True/False
    },
    "inner": {
        "args": {
            "input1": "tensor",
            "input2": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.CELU"] = {
    "args": {},
    "kwargs": {
        "alpha": "float", # could be number, but float is more precise
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.CTCLoss"] = {
    "args": {
        "blank": "integer",  # Could be float as well
    },
    "kwargs": {
        "reduction": "string",
        "zero_infinity": "boolean",
    },
    "inner": {
        "args": {
            "log_probs": "tensor",
            "targets": "tensor",
            "input_lengths": "tensor_list", # Could also be tensor or tuple
            "target_lengths": "tensor_list" # Could also be tensor or tuple
        },
        "kwargs": {}
    }
}
signatures["torch.nn.CircularPad1d_1"] = {
    "args": {
        "padding": "integer" # could also be tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.CircularPad1d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad1d_1"] = {
    "args": {
        "padding": "integer",
        "value": "float"  # Could also be a tensor
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad1d_2"] = {
    "args": {
        "padding": "tuple",
        "value": "float"  # Could also be a tensor
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad2d"] = {
    "args": {
        "padding": "tuple"  # can also be an integer
    },
    "kwargs": {
        "value": "float"
    },
    "inner": {}
}
signatures["torch.nn.ConstantPad3d_1"] = {
    "args": {
        "padding": "integer" # could also be tuple
    },
    "kwargs": {
        "value": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad3d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {
        "value": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Conv1d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer",  # or tuple
        "stride": "integer",  # or tuple
    },
    "kwargs": {
        "padding": "integer",  # or tuple or string
        "dilation": "integer",  # or tuple
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype",
    },
    "inner": {},
}
signatures["torch.nn.Conv2d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple",  # can be int or tuple
        "stride": "tuple",  # can be int or tuple
        "padding": "tuple",  # can be int, tuple or string
        "dilation": "tuple",  # can be int or tuple
        "groups": "integer",
        "bias": "boolean"
    },
    "kwargs": {
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.Conv3d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple",  # can also be int
        "stride": "tuple",  # can also be int
        "padding": "tuple",  # can also be int or string
        "dilation": "tuple",  # can also be int
        "groups": "integer",
        "bias": "boolean"
    },
    "kwargs": {
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.ConvTranspose1d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer",  # or tuple
        "stride": "integer",  # or tuple
        "padding": "integer",  # or tuple
        "output_padding": "integer",  # or tuple
        "groups": "integer",
        "bias": "boolean",
        "dilation": "integer",  # or tuple
    },
    "kwargs": {
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.ConvTranspose2d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple",  # or integer
        "stride": "tuple",  # or integer
        "padding": "tuple",  # or integer
        "output_padding": "tuple",  # or integer
        "groups": "integer",
        "bias": "boolean",
        "dilation": "tuple",  # or integer
    },
    "kwargs": {
        "padding_mode": "string",
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {
            "output_size": "list",  # or None
        },
    },
}
signatures["torch.nn.ConvTranspose3d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple",  # or integer
        "stride": "tuple",  # or integer
        "padding": "tuple",  # or integer
        "output_padding": "tuple",  # or integer
        "groups": "integer",
        "bias": "boolean",
        "dilation": "tuple",  # or integer
    },
    "kwargs": {
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.CosineEmbeddingLoss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "target": "tensor"  # could also be an empty tensor ()
    },
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.nn.CosineSimilarity"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "dim": "integer", # could also be a tuple
        "eps": "float"
    },
    "inner": {}
}
signatures["torch.nn.CrossEntropyLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"  # Could also be a list of tensors, but documentation is ambiguous.
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",  # Deprecated
        "ignore_index": "integer",
        "reduce": "boolean",  # Deprecated
        "reduction": "string",
        "label_smoothing": "float"
    },
    "inner": {}
}
signatures["torch.nn.Dropout"] = {
    "args": {},
    "kwargs": {
        "p": "float", # could also be interpreted as a number
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor" #Input can be of any shape
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Dropout2d"] = {
    "args": {},
    "kwargs": {
        "p": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"  # Could also be tensor_list, but the Shape section indicates a single tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Dropout3d"] = {
    "args": {
        # p could also be an integer, but float seems more likely
        "p": "float"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"  # Input shape is (N,C,D,H,W) or (C,D,H,W)
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ELU"] = {
    "args": {},
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer",
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float", # could be integer as well
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean",
    },
    "inner": {},
}

signatures["torch.nn.Embedding.from_pretrained"] = {
    "args": {
        "embeddings": "tensor",
    },
    "kwargs": {
        "freeze": "boolean",
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float", # could be integer
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.EmbeddingBag"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer",
        "input": "tensor",
        "offsets": "tensor",
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "include_last_offset": "boolean",
        "padding_idx": "integer",
        "per_sample_weights": "tensor",  # Could also be None
    },
    "inner": {},
}
signatures["torch.nn.FeatureAlphaDropout"] = {
    "args": {},
    "kwargs": {
        "p": "float", # could also be interpreted as a probability
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Flatten"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "start_dim": "integer",  # Could also be None
        "end_dim": "integer"      # Could also be None
    },
    "inner": {},
}
signatures["torch.nn.Fold"] = {
    "args": {
        "output_size": "tuple",  # Could be int or tuple
        "kernel_size": "tuple",  # Could be int or tuple
    },
    "kwargs": {
        "dilation": "tuple",  # Could be int or tuple, default 1
        "padding": "tuple",  # Could be int or tuple, default 0
        "stride": "tuple",  # Could be int or tuple, default 1
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FractionalMaxPool2d"] = {
    "args": {
        "kernel_size": "union[integer, tuple]",  # Could be int or tuple of ints
        "output_size": "union[integer, tuple]",  # Could be int or tuple of ints
        "output_ratio": "union[float, tuple]",  # Could be float or tuple of floats
    },
    "kwargs": {
        "return_indices": "boolean",  # Default: False
    },
    "inner": {},
}
signatures["torch.nn.FractionalMaxPool3d"] = {
    "args": {
        "kernel_size": "union[integer, tuple]",  # Could be int or tuple of ints
        "output_size": "union[integer, tuple]",  # Could be int or tuple of ints
        "output_ratio": "union[float, tuple]",  # Could be float or tuple of floats
        "return_indices": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.GELU"] = {
    "args": {},
    "kwargs": {
        "approximate": "string" # could also be boolean, but string seems most accurate
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.GLU"] = {
    "args": {
        "input": "tensor"  # Should it be tensor_list? Documentation doesn't specify.
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.nn.GRU"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
        "num_layers": "integer",
        "h_0": "tensor",  # Could also be tensor_list
    },
    "kwargs": {
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "dtype": "dtype", # May be a string too
    },
    "inner": {},
}
signatures["torch.nn.GRUCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
    },
    "kwargs": {
        "bias": "boolean", # could also be None
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hidden": "tensor" # could be omitted
        },
        "kwargs": {}
    }
}
signatures["torch.nn.GaussianNLLLoss"] = {
    "args": {},
    "kwargs": {
        "full": "boolean",
        "eps": "float",
        "reduction": "string"  # Could also be 'none', 'mean', or 'sum'
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor",
            "var": "tensor"  # Could be scalar too
        },
        "kwargs": {}
    }
}
signatures["torch.nn.GroupNorm"] = {
    "args": {
        "num_groups": "integer",
        "num_channels": "integer",
    },
    "kwargs": {
        "eps": "float", # could be a tensor too
        "affine": "boolean",
        "dtype": "dtype" # unsure, could be tensor as well
    },
    "inner": {},
}
signatures["torch.nn.Hardshrink"] = {
    "args": {
        "lambd": "float" # could also be a tensor
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Hardsigmoid"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean" # could be None too, but documentation does not state it
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Hardswish"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"  # Could also be None, but boolean is more restrictive and documented as default
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Hardtanh"] = {
    "args": {
        "min_val": "float",
        "max_val": "float"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.HingeEmbeddingLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",  # Deprecated, should be float
        "reduce": "boolean",  # Deprecated, should be boolean
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.HuberLoss"] = {
    "args": {},
    "kwargs": {
        "reduction": "string",  # Could be 'none', 'mean', or 'sum'
        "delta": "float"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Identity"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"  # input can be any shape tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.InstanceNorm1d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"  # Could also be None
    },
    "inner": {},
}
signatures["torch.nn.InstanceNorm2d"] = {
    "args": {
        "num_features": "integer"  # Could also be a tuple, but documentation only lists int
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype" # Assuming dtype is a valid type
    },
    "inner": {},
}
signatures["torch.nn.InstanceNorm3d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"  # Could also be None
    },
    "inner": {}
}
signatures["torch.nn.KLDivLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",  # Deprecated, should be float
        "reduce": "boolean",  # Deprecated, should be boolean
        "reduction": "string",
        "log_target": "boolean"
    },
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.nn.L1Loss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean", # could be deprecated
        "reduce": "boolean", # could be deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LPPool1d"] = {
    "args": {
        "norm_type": "integer",  # could be float as well, but integer seems more likely based on description
        "kernel_size": "union[int, tuple]",
        "stride": "union[int, tuple]"
    },
    "kwargs": {
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LPPool2d"] = {
    "args": {
        "norm_type": "float",  # could be int as well
        "kernel_size": "tuple", # can be integer as well
    },
    "kwargs": {
        "stride": "tuple", # can be integer as well
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LSTM"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
        "num_layers": "integer",
    },
    "kwargs": {
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "proj_size": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.LSTMCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "bias": "boolean",  # Could also be None
        "dtype": "dtype"  # dtype is not explicitly mentioned but is a common kwarg
    },
    "inner": {
        "args": {
            "input": "tensor",  # Can also be a single tensor
            "h_0": "tensor",  # Can also be a single tensor
            "c_0": "tensor"  # Can also be a single tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LayerNorm"] = {
    "args": {
        "normalized_shape": "list",  # Could also be int or torch.Size
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "bias": "boolean",
        "dtype": "dtype" # might also be None
    },
    "inner": {},
}
signatures["torch.nn.LazyBatchNorm1d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Can be None, but float is more general
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.LazyBatchNorm2d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Can be None, but float is more general
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.LazyInstanceNorm1d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",  # could be a tensor
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.LazyInstanceNorm2d"] = {
    "args": {},
    "kwargs": {
        "eps": "float", # could be tensor as well
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.LeakyReLU"] = {
    "args": {},
    "kwargs": {
        "negative_slope": "float", # Could also be integer
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Linear"] = {
    "args": {
        "in_features": "integer",
        "out_features": "integer"
    },
    "kwargs": {
        "bias": "boolean" # could also be None, but boolean is the main type
    },
    "inner": {
        "args": {
            "input": "tensor" # from Shape: Input: (,Hin)
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LocalResponseNorm"] = {
    "args": {
        "size": "integer",
        "input": "tensor" # The documentation mentions input under forward pass but also as a parameter
    },
    "kwargs": {
        "alpha": "float",
        "beta": "float",
        "k": "float"
    },
    "inner": {}
}
signatures["torch.nn.LogSigmoid"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"  # input can be any number of dimensions, so tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LogSoftmax"] = {
    "args": {
        "dim": "integer" # Could also be None, but integer is more specific
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MSELoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",  # Could be deprecated, consider removing
        "reduce": "boolean",  # Could be deprecated, consider removing
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MarginRankingLoss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "target": "tensor"  # could also be a 0D tensor
    },
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.MaxPool1d"] = {
    "args": {
        "kernel_size": "integer",  # could also be tuple
        "stride": "integer",  # could also be tuple, defaults to kernel_size
        "padding": "integer",  # could also be tuple
        "dilation": "integer",  # could also be tuple
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MaxPool2d"] = {
    "args": {
        "kernel_size": "tuple",  # could also be integer
        "stride": "tuple",  # could also be integer, defaults to kernel_size
        "padding": "tuple",  # could also be integer, defaults to 0
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_1"] = {
    "args": {
        "kernel_size": "integer",
        "stride": "tuple",  # could also be integer, defaults to kernel_size
        "padding": "tuple",  # could also be integer, defaults to 0
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_2"] = {
    "args": {
        "kernel_size": "integer",
        "stride": "integer",  # could also be integer, defaults to kernel_size
        "padding": "tuple",  # could also be integer, defaults to 0
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_3"] = {
    "args": {
        "kernel_size": "tuple",  # could also be integer
        "stride": "integer",  # could also be integer, defaults to kernel_size
        "padding": "tuple",  # could also be integer, defaults to 0
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_4"] = {
    "args": {
        "kernel_size": "tuple",  # could also be integer
        "stride": "tuple",  # could also be integer, defaults to kernel_size
        "padding": "integer",  # could also be integer, defaults to 0
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_5"] = {
    "args": {
        "kernel_size": "integer",  # could also be integer
        "stride": "tuple",  # could also be integer, defaults to kernel_size
        "padding": "integer",  # could also be integer, defaults to 0
        "dilation": "integer",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool3d"] = {
    "args": {
        "kernel_size": "tuple",  # or integer
        "stride": "tuple",  # or integer, defaults to kernel_size
        "padding": "tuple",  # or integer, defaults to 0
        "dilation": "tuple",  # or integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxUnpool1d"] = {
    "args": {
        "kernel_size": "integer",  # or tuple
        "stride": "integer",  # or tuple
        "padding": "integer"  # or tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor",
        },
        "kwargs": {
            "output_size": "tuple"  # or None
        }
    }
}
signatures["torch.nn.MaxUnpool2d"] = {
    "args": {
        "kernel_size": "tuple",  # Could also be int
        "stride": "tuple",  # Could also be int, defaults to kernel_size
        "padding": "integer"  # Could also be tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "tuple"  # Can also be None
        }
    }
}
signatures["torch.nn.MaxUnpool3d"] = {
    "args": {
        "kernel_size": "tuple",  # Can also be an integer
        "stride": "tuple",  # Can also be an integer, defaults to kernel_size
        "padding": "integer"  # Can also be a tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "tuple"  # optional
        }
    }
}
signatures["torch.nn.Mish"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"  # Could also be None, but boolean seems more appropriate
    },
    "inner": {},
}
signatures["torch.nn.MultiLabelMarginLoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MultiLabelSoftMarginLoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",  # could also be None
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MultiMarginLoss"] = {
    "args": {
        "p": "integer",
        "margin": "float",
    },
    "kwargs": {
        "weight": "tensor",  # Could be None, but treating as tensor for simplicity
        "size_average": "boolean",  # Deprecated
        "reduce": "boolean",  # Deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"  # or integer
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MultiheadAttention"] = {
    "args": {
        "embed_dim": "integer",
        "num_heads": "integer"
    },
    "kwargs": {
        "dropout": "float",
        "bias": "boolean",
        "add_bias_kv": "boolean",
        "add_zero_attn": "boolean",
        "kdim": "integer", # could be None
        "vdim": "integer", # could be None
        "batch_first": "boolean"
    },
    "inner": {
        "args": {
            "query": "tensor",
            "key": "tensor",
            "value": "tensor"
        },
        "kwargs": {
            "key_padding_mask": "tensor", # could be None
            "need_weights": "boolean",
            "attn_mask": "tensor", # could be None
            "average_attn_weights": "boolean",
            "is_causal": "boolean"
        }
    }
}
signatures["torch.nn.NLLLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"  # Could also be a list of tensors
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",  # Deprecated
        "ignore_index": "integer",
        "reduce": "boolean",  # Deprecated
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.PReLU"] = {
    "args": {
        "num_parameters": "integer", # Could also be a tensor, but documentation specifies an integer
    },
    "kwargs": {
        "init": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.PairwiseDistance"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "p": "float",  # Could also be integer
        "eps": "float",
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.Parameter"] = {
    "args": {
        "data": "tensor" # could also be tensor_list
    },
    "kwargs": {
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.ParameterList"] = {
    "args": {
        "values": "list"  # could also be an iterable
    },
    "kwargs": {},
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.nn.ParameterList_append"] = {
    "args": {
        "value": "tensor" # Could be Any, but likely tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.ParameterList_extend"] = {
    "args": {
        "values": "list" # could also be an iterable
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.ParameterList_extra_repr"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.PixelShuffle"] = {
    "args": {
        "upscale_factor": "integer"  # Should it be long? Assuming integer for now
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.PixelUnshuffle"] = {
    "args": {
        "downscale_factor": "integer"  # Should it be long? Assuming integer for now
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.PoissonNLLLoss"] = {
    "args": {},
    "kwargs": {
        "log_input": "boolean",  # could be interpreted as tensor as well
        "full": "boolean",
        "size_average": "boolean", # Deprecated
        "eps": "float",
        "reduce": "boolean", # Deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {
            "log_input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RMSNorm"] = {
    "args": {
        "normalized_shape": "list"  # Could also be an integer
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean"
    },
    "inner": {
        "args": {
            "x": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RNN"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
        "num_layers": "integer",
    },
    "kwargs": {
        "nonlinearity": "string",  # could be 'tanh' or 'relu'
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "device": "string", # skipped
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "input": "tensor",  # or PackedSequence
            "hx": "tensor",  # or None
        },
        "kwargs": {
            "batch_first": "boolean",
        },
    },
}
signatures["torch.nn.RNNCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "bias": "boolean",
        "nonlinearity": "string" # could also be a callable
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hidden": "tensor" # or could be None, in which case it defaults to zero
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RReLU"] = {
    "args": {},
    "kwargs": {
        "lower": "float",
        "upper": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"  # Input is a tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReLU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"  # Could also be None, but boolean is more restrictive.
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReLU6"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"  # Could also be True/False
    },
    "inner": {},
}
signatures["torch.nn.ReflectionPad1d"] = {
    "args": {
        "padding": "integer"  # or tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad2d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # could also be tensor_list
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad2d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # could also be tensor_list
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad3d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # could also be tensor_list
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad3d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # could also be tensor_list
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad1d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"  # should be tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad1d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"  # should be tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad2d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # Could also be a tensor list
        },
        "kwargs": {}
    }
}

signatures["torch.nn.ReplicationPad2d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # Could also be a tensor list
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad3d_1"] = {
    "args": {
        "padding": "integer" # could also be tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad3d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.SELU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.SiLU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"  # Could also be None, but boolean seems more appropriate
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Sigmoid"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"  # The input to the sigmoid function
        },
        "kwargs": {}
    }
}
signatures["torch.nn.SmoothL1Loss"] = {
    "args": {},
    "kwargs": {
        "beta": "float",  # could also be integer
        "reduction": "string",
        "size_average": "boolean", # deprecated
        "reduce": "boolean" # deprecated
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.SoftMarginLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",  # could be deprecated
        "reduce": "boolean",  # could be deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.nn.Softmax"] = {
    "args": {},
    "kwargs": {
        "dim": "integer" # Could also be None, but integer is more specific
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softmax2d"] = {
    "args": {
        "input": "tensor"  # could also be tensor_list
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.Softmin"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {}
}
signatures["torch.nn.Softplus"] = {
    "args": {},
    "kwargs": {
        "beta": "float", # could also be integer
        "threshold": "float" # could also be integer
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softshrink"] = {
    "args": {
        "lambd": "float" # could also be a tensor
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softsign"] = {
    "args": {
        "input": "tensor"  # The documentation says Input: ()(*)(), where * means any number of dimensions. So tensor should be fine.
    },
    "kwargs": {},
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.nn.SyncBatchNorm"] = {
    "args": {
        "num_features": "integer",
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # can be None
        "affine": "boolean",
        "track_running_stats": "boolean",
        "process_group": "list",  # Assuming it's a process group which is a list
        "device": "string", # Not included as per instructions
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Tanh"] = {
    "args": {
        "input": "tensor"  # could also be tensor_list, but documentation mentions element-wise operation, which applies to tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.Tanhshrink"] = {
    "args": {
        "input": "tensor"  # Should it be tensor_list? Documentation does not specify.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.Threshold"] = {
    "args": {
        "threshold": "float",
        "value": "float"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor" # could also be tensor_list
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Transformer"] = {
    "args": {
        "src": "tensor",
        "tgt": "tensor"
    },
    "kwargs": {
        "d_model": "integer",
        "nhead": "integer",
        "num_encoder_layers": "integer",
        "num_decoder_layers": "integer",
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string",  # Could also be a callable
        "custom_encoder": "object",  # Not sure about exact type
        "custom_decoder": "object",  # Not sure about exact type
        "layer_norm_eps": "float",
        "batch_first": "boolean",
        "norm_first": "boolean",
        "bias": "boolean",
        "src_mask": "tensor",
        "tgt_mask": "tensor",
        "memory_mask": "tensor",
        "src_key_padding_mask": "tensor",
        "tgt_key_padding_mask": "tensor",
        "memory_key_padding_mask": "tensor",
        "src_is_causal": "boolean",
        "tgt_is_causal": "boolean",
        "memory_is_causal": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.Transformer_generate_square_subsequent_mask"] = {
    "args": {
        "sz": "integer"
    },
    "kwargs": {
        "device": "object", #Should be device but it is not on allowed list
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.TransformerDecoderLayer"] = {
    "args": {
        "d_model": "integer",
        "nhead": "integer"
    },
    "kwargs": {
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string", # could also be callable
        "layer_norm_eps": "float",
        "batch_first": "boolean",
        "norm_first": "boolean",
        "bias": "boolean"
    },
    "inner": {
        "args": {
            "tgt": "tensor",
            "memory": "tensor"
        },
        "kwargs": {
            "tgt_mask": "tensor", # optional
            "memory_mask": "tensor", # optional
            "tgt_key_padding_mask": "tensor", # optional
            "memory_key_padding_mask": "tensor", # optional
            "tgt_is_causal": "boolean",
            "memory_is_causal": "boolean"
        }
    }
}
signatures["torch.nn.TransformerEncoderLayer"] = {
    "args": {
        "d_model": "integer",
        "nhead": "integer"
    },
    "kwargs": {
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string", # could also be callable
        "layer_norm_eps": "float",
        "batch_first": "boolean",
        "norm_first": "boolean",
        "bias": "boolean"
    },
    "inner": {
        "args": {
            "src": "tensor" # or NestedTensor
        },
        "kwargs": {
            "src_mask": "tensor",
            "src_key_padding_mask": "tensor",
            "is_causal": "boolean"
        }
    }
}
signatures["torch.nn.TripletMarginLoss"] = {
    "args": {
        "anchor": "tensor",
        "positive": "tensor",
        "negative": "tensor"
    },
    "kwargs": {
        "margin": "float", # could also be integer
        "p": "integer",
        "eps": "float",
        "swap": "boolean",
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.Unflatten"] = {
    "args": {
        "dim": "integer",
        "unflattened_size": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Unfold"] = {
    "args": {
        "kernel_size": "tuple",  # can be int or tuple
        "input": "tensor"
    },
    "kwargs": {
        "dilation": "tuple",  # can be int or tuple, default 1
        "padding": "tuple",  # can be int or tuple, default 0
        "stride": "tuple"  # can be int or tuple, default 1
    },
    "inner": {}
}
signatures["torch.nn.UninitializedBuffer"] = {
    "args": {
        "size": "tuple" # Could also be an integer, but tuple is more general.
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.UninitializedParameter"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.Upsample"] = {
    "args": {
        "size": "tuple",  # Could be int or tuple
        "scale_factor": "float",  # Could be float or tuple
    },
    "kwargs": {
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.ZeroPad1d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # Could also be tensor_list, but examples show only a single tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ZeroPad1d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # Could also be tensor_list, but examples show only a single tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ZeroPad2d_1"] = {
    "args": {
        "padding": "integer"  # Could also be a tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ZeroPad2d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.functional.adaptive_avg_pool1d"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "integer", # Could also be None
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool2d"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # or integer, but tuple seems more general
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool3d"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # could also be integer
    },
    "kwargs": {
        "return_indices": "boolean" # Might be useful to check if it can be other types
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool1d"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"  # could also be a tuple, see below
    },
    "kwargs": {
        "dim": "integer",
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.adaptive_max_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {
        "dim": "integer",
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.adaptive_max_pool2d"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # could also be integer
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.adaptive_max_pool3d"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # could also be integer
    },
    "kwargs": {
        "return_indices": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.functional.affine_grid"] = {
    "args": {
        "theta": "tensor",
        "size": "tuple"  # could also be list
    },
    "kwargs": {
        "dtype": "dtype"  # could be None
    },
    "inner": {}
}
signatures["torch.nn.functional.alpha_dropout"] = {
    "args": {
        "input": "tensor",
        "dropout_prob": "float",
        "train": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.avg_pool1d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
    },
    "kwargs": {
        "divisor": "integer",  # Should it be float?
        "out": "tensor",
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool2d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",  # or tuple
        "stride": "integer",  # or tuple
        "padding": "integer",  # or tuple
        "dilation": "integer",  # or tuple
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "count_include_pad": "boolean",
        "divisor": "float"
    },
    "inner": {}
}
signatures["torch.nn.functional.avg_pool3d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",  # Could also be an integer
    },
    "kwargs": {
        "stride": "tuple",  # Could also be an integer
        "padding": "tuple",  # Could also be an integer
        "dilation": "tuple",  # Could also be an integer
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.batch_norm"] = {
    "args": {
        "input": "tensor",
        "running_mean": "tensor",
        "running_var": "tensor",
        "weight": "tensor",
        "bias": "tensor",
    },
    "kwargs": {
        "training": "boolean",
        "momentum": "float",
        "eps": "float",
        "moving_average": "float", # Could also be boolean, using float since it's a numerical value
        "affine": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.functional.bilinear"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",  # Could be None, but documented as tensor
    },
    "inner": {},
}
signatures["torch.nn.functional.binary_cross_entropy"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",  # could be None
        "size_average": "boolean", # deprecated
        "reduce": "boolean",
        "reduction": "string",
        "log_softmax": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.binary_cross_entropy_with_logits"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor", # Could be None
        "size_average": "boolean", # Deprecated
        "reduce": "boolean",
        "pos_weight": "tensor" # Could be None
    },
    "inner": {},
}
signatures["torch.nn.functional.celu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"  # Could also be None, but boolean is more likely given documentation
    },
    "inner": {}
}
signatures["torch.nn.functional.conv1d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",  # or None
        "stride": "integer",  # or tuple
        "padding": "string",  # or integer or tuple
        "dilation": "integer",  # or tuple
        "groups": "integer",
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",  # can be tuple
        "padding": "string",  # can be integer or tuple
        "dilation": "integer",  # can be tuple
        "groups": "integer",
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # can also be tuple
        "padding": "string", # can also be integer or tuple
        "dilation": "integer", # can also be tuple
        "groups": "integer",
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose1d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor",
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "output_size": "integer", # Could be a tuple, adding separate signature if needed
        "groups": "integer",
        "dtype": "dtype",
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor",  # could be None
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "dilation": "tuple"
    },
    "kwargs": {
        "groups": "integer",
        "dtype": "dtype"  # might be None
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose3d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor",
        "stride": "tuple", # could also be integer
        "padding": "tuple", # could also be integer
        "dilation": "tuple", # could also be integer
        "groups": "integer",
    },
    "kwargs": {
        "output_size": "tuple", # or integer
    },
    "inner": {},
}
signatures["torch.nn.functional.cosine_embedding_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
    },
    "kwargs": {
        "margin": "float",  # Could also be a tensor, but float seems more common
        "reduction": "string" # Should be string, options are 'mean', 'sum', 'none'
    },
    "inner": {},
}
signatures["torch.nn.functional.cosine_similarity"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
    },
    "kwargs": {
        "dim": "integer",  # could also be -1
        "eps": "float",
    },
    "inner": {},
}
signatures["torch.nn.functional.cross_entropy"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
    },
    "kwargs": {
        "weight": "tensor",  # could be None
        "size_average": "boolean", # deprecated
        "ignore_index": "integer",
        "reduction": "string",
        "label_smoothing": "float", # could be 0.0
    },
    "inner": {},
}
signatures["torch.nn.functional.cross_entropy_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
        "logits": "tensor"
    },
    "kwargs": {
        "weight": "tensor",  # could be None
        "size_average": "boolean", # deprecated
        "ignore_index": "integer",
        "reduction": "string",
        "label_smoothing": "float", # could be 0.0
    },
    "inner": {},
}
signatures["torch.nn.functional.ctc_loss"] = {
    "args": {
        "input": "tensor",
        "targets": "tensor",
        "input_lengths": "tensor",
        "target_lengths": "tensor"
    },
    "kwargs": {
        "blank": "integer", # could be float as well
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.functional.dropout"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float",  # Could also be interpreted as a number
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.dropout2d"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.functional.dropout3d"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.functional.elu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean" # could also be None, but boolean is more appropriate based on description
    },
    "inner": {},
}
signatures["torch.nn.functional.embedding"] = {
    "args": {
        "input": "tensor",  # LongTensor
        "weight": "tensor"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.embedding_bag"] = {
    "args": {
        "embedding_matrix": "tensor",
        "indices": "tensor",
        "offsets": "tensor",
    },
    "kwargs": {
        "max_norm": "float",  # Could be None
        "norm_type": "integer",  # 2 or 1
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.feature_alpha_dropout"] = {
    "args": {
        "input": "tensor",
        "alpha": "float",
        "p": "float"
    },
    "kwargs": {
        "training": "boolean",  # Could also be an integer (0 or 1)
    },
    "inner": {},
}
signatures["torch.nn.functional.fold"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple", # or list? Assuming tuple as it's size.
        "kernel_size": "tuple", # or list? Assuming tuple as it's size.
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.fractional_max_pool2d"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # Could also be integer, creating another signature for that
    },
    "kwargs": {
        "kernel_size": "tuple",  # or integer
        "stride": "tuple",  # or integer
        "padding": "tuple",  # or integer
        "dilation": "tuple",  # or integer
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.fractional_max_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "kernel_size": "tuple",  # or integer
        "stride": "tuple",  # or integer
        "padding": "tuple",  # or integer
        "dilation": "tuple",  # or integer
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.fractional_max_pool2d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {
        "kernel_size": "integer",
        "stride": "tuple",  # or integer
        "padding": "tuple",  # or integer
        "dilation": "tuple",  # or integer
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.fractional_max_pool2d_3"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "kernel_size": "integer",
        "stride": "tuple",  # or integer
        "padding": "tuple",  # or integer
        "dilation": "tuple",  # or integer
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.gelu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "approximate": "string"  # Could be 'none' or 'tanh'
    },
    "inner": {}
}
signatures["torch.nn.functional.glu"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # could also be None
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.grid_sample"] = {
    "args": {
        "input": "tensor",
        "grid": "tensor",
        "mode": "string",
        "padding_mode": "string",
        "align_corners": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.group_norm"] = {
    "args": {
        "input": "tensor",
        "num_groups": "integer",
        "eps": "float",
        "affine": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.gumbel_softmax"] = {
    "args": {
        "input": "tensor",
        "tau": "float",
        "hard": "boolean"
    },
    "kwargs": {
        "dim": "integer",  # Could be -1 as well. Choosing integer.
        "eps": "float"  # small value to avoid log(0)
    },
    "inner": {}
}
signatures["torch.nn.functional.hardshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float" # could also be an integer
    },
    "inner": {},
}
signatures["torch.nn.functional.hardsigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.hardswish"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None, but boolean seems more appropriate
    },
    "inner": {}
}
signatures["torch.nn.functional.hardtanh"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "min_val": "float", # could also be tensor
        "max_val": "float", # could also be tensor
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.hinge_embedding_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float", # Could also be integer, but float seems more common.
        "dist_function": "string" # Should be a callable, but string is the best fit.
    },
    "inner": {}
}
signatures["torch.nn.functional.huber_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "delta": "float", # Could also be a tensor
        "reduction": "string" # Could be 'none', 'mean', 'sum', etc.
    },
    "inner": {},
}
signatures["torch.nn.functional.instance_norm"] = {
    "args": {
        "input": "tensor",
        "running_mean": "tensor",
        "running_var": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "eps": "float",  # Could be a float or a Python scalar.
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.interpolate"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "size": "tuple",  # Could be int or tuple of ints
        "scale_factor": "float",  # Could be float or tuple of floats
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean",
        "antialias": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.functional.kl_div"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
    },
    "kwargs": {
        "log_target": "boolean",  # Could be tensor as well, depending on what's being passed.
        "reduction": "string",  # Should be "none", "sum", or "batchmean"
        "log_input": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.l1_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string", # Could be 'none', 'mean', 'sum'
        "size_average": "boolean" # deprecated
    },
    "inner": {}
}
signatures["torch.nn.functional.layer_norm"] = {
    "args": {
        "input": "tensor",
        "normalized_shape": "tuple",  # Could also be a list of integers
    },
    "kwargs": {
        "weight": "tensor",
        "bias": "tensor",
        "eps": "float",
    },
    "inner": {},
}
signatures["torch.nn.functional.leaky_relu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "negative_slope": "float"  # Could also be an integer, but float seems more general
    },
    "inner": {}
}
signatures["torch.nn.functional.linear"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor"  # or None, but defaulting to tensor for simplicity
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.local_response_norm"] = {
    "args": {
        "input": "tensor",
        "size": "integer",
        "alpha": "float",
        "beta": "float",
        "eps": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.log_softmax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype" # could be torch.dtype as well
    },
    "inner": {}
}
signatures["torch.nn.functional.logsigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.lp_pool1d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "norm_type": "integer" # Could also be float, but integer seems more appropriate given the documentation
    },
    "kwargs": {
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.lp_pool2d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer", # could also be tuple of integers
        "stride": "integer", # could also be tuple of integers
        "padding": "integer", # could also be tuple of integers or string
        "norm_type": "integer"
    },
    "kwargs": {
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.margin_ranking_loss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "label": "tensor"  # could be integer as well
    },
    "kwargs": {
        "margin": "float",
        "reduction": "string"  # could also be integer
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool1d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"  # Could be optional, but documented as the desired output type
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer", # could also be tuple
        "stride": "integer", # could also be tuple
        "padding": "integer", # could also be tuple
        "dilation": "integer", # could also be tuple
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_3"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_4"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_5"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_6"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_7"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_8"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_9"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_10"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_11"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple" # deprecated, keeping for completeness
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool3d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer", # or tuple/list
        "stride": "integer", # or tuple/list
        "padding": "integer", # or tuple/list
        "dilation": "integer", # or tuple/list
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_3"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_4"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_5"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_6"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_7"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_8"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_9"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_10"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_11"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_12"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_13"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_14"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_pool3d_15"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.functional.max_unpool1d"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer" # Could be integer or tuple, using integer for simplicity
    },
    "inner": {},
}
signatures["torch.nn.functional.max_unpool2d"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "kernel_size": "tuple"  # or integer, but tuple seems more common
    },
    "kwargs": {
        "stride": "tuple", # or integer
        "padding": "tuple", # or integer
        "dilation": "tuple" # or integer
    },
    "inner": {},
}
signatures["torch.nn.functional.max_unpool3d"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "kernel_size": "tuple" # Could also be an integer
    },
    "kwargs": {
        "stride": "tuple", # Could also be an integer
        "padding": "tuple", # Could also be an integer
        "dilation": "tuple" # Could also be an integer
    },
    "inner": {},
}
signatures["torch.nn.functional.mish"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None, but boolean seems more appropriate
    },
    "inner": {}
}
signatures["torch.nn.functional.mse_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string",  # could be 'none', 'mean', or 'sum'
        "size_average": "boolean", # deprecated, kept for compatibility
        "reduce": "boolean", # deprecated, kept for compatibility
    },
    "inner": {},
}
signatures["torch.nn.functional.multi_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
    },
    "kwargs": {
        "p": "float",  # Could also be integer, but float seems more general
        "margin": "float",
        "head": "integer",
    },
    "inner": {},
}
signatures["torch.nn.functional.multilabel_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",  # Should it be integer as well?
        "reduction": "string"  # Could be "mean", "sum", or "none"
    },
    "inner": {}
}
signatures["torch.nn.functional.multilabel_soft_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",  # could be None
        "ignore_index": "integer", # could be -100
        "reduction": "string" # could be "none", "mean", or "sum"
    },
    "inner": {}
}
signatures["torch.nn.functional.nll_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
    },
    "kwargs": {
        "weight": "tensor", # Could be None
        "ignore_index": "integer", # Could be -100
        "reduction": "string", # Could be 'mean', 'sum', or 'none'
        "log_softmax": "boolean", # Added to cover the case where input is log probabilities
    },
    "inner": {},
}
signatures["torch.nn.functional.normalize"] = {
    "args": {
        "input": "tensor",
        "p": "float",
        "dim": "integer",  # Can also be a tuple of integers
        "eps": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.functional.one_hot"] = {
    "args": {
        "input": "tensor",
        "num_classes": "integer"
    },
    "kwargs": {
        "dtype": "dtype" # Could be torch.dtype instead
    },
    "inner": {}
}
signatures["torch.nn.functional.pad"] = {
    "args": {
        "input": "tensor",
        "pad": "tuple",  # can also be a list
    },
    "kwargs": {
        "mode": "string",
        "value": "float"  # could be a dtype as well
    },
    "inner": {},
}
signatures["torch.nn.functional.pairwise_distance"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor"
    },
    "kwargs": {
        "eps": "float",  # Could also be integer
        "metric": "string",
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.pdist"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float",  # Could also be integer
    },
    "inner": {},
}
signatures["torch.nn.functional.pixel_shuffle"] = {
    "args": {
        "input": "tensor",
        "upscale_factor": "integer"  # Could also be a tuple of integers
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.pixel_unshuffle"] = {
    "args": {
        "input": "tensor",
        "r": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.poisson_nll_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "log_prob": "boolean",  # Could be tensor as well
        "reduction": "string"  # Could also be None
    },
    "inner": {}
}
signatures["torch.nn.functional.prelu"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"  # could also be a scalar float
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.relu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None, but boolean seems more appropriate
    },
    "inner": {}
}
signatures["torch.nn.functional.relu6"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None
    },
    "inner": {},
}
signatures["torch.nn.functional.rrelu"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "lower": "float",  # could also be integer
        "upper": "float",  # could also be integer
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.selu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None
    },
    "inner": {}
}
signatures["torch.nn.functional.silu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None, but boolean is more common for inplace operations
    },
    "inner": {}
}
signatures["torch.nn.functional.smooth_l1_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "beta": "float",  # could also be tensor
        "reduction": "string"  # could be "mean", "sum", or "none"
    },
    "inner": {}
}
signatures["torch.nn.functional.soft_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string"  # Should it be enum?
    },
    "inner": {}
}
signatures["torch.nn.functional.softmax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype"  # could also be None
    },
    "inner": {}
}
signatures["torch.nn.functional.softmin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.functional.softplus"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "beta": "float",  # could also be integer
        "threshold": "float"  # could also be integer
    },
    "inner": {}
}
signatures["torch.nn.functional.softshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float" # could also be an integer
    },
    "inner": {},
}
signatures["torch.nn.functional.softsign"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.tanhshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.threshold"] = {
    "args": {
        "input": "tensor",
        "threshold": "float", # could also be integer
        "value": "float" # could also be integer
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.triplet_margin_loss"] = {
    "args": {
        "anchor": "tensor",
        "positive": "tensor",
        "negative": "tensor",
    },
    "kwargs": {
        "margin": "float",  # could also be a tensor
        "p": "float",
        "eps": "float",
        "reduction": "string", # Should this be enum?
    },
    "inner": {},
}
signatures["torch.nn.functional.unfold"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple", # could also be an integer
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.calculate_gain"] = {
    "args": {
        "fan_in": "integer",  # Could also be a tensor, but integer seems more common
        "fan_out": "integer",  # Could also be a tensor, but integer seems more common
        "distribution": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.constant_"] = {
    "args": {
        "tensor": "tensor",
        "val": "float" # Could also be integer
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.dirac_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "with_offset": "boolean" # Could be integer as well.
    },
    "inner": {},
}
signatures["torch.nn.init.eye_"] = {
    "args": {
        "n": "integer" # could also be a tuple, but integer is the primary type
    },
    "kwargs": {
        "m": "integer", # could also be a tuple
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.init.kaiming_normal_"] = {
    "args": {
        "tensor": "tensor",
        "mode": "string"  # Could also be integer, representing mode
    },
    "kwargs": {
        "distribution": "string", # Could be "fan_in" or "fan_out"
        "a": "float",
        "nonlinearity": "string"
    },
    "inner": {}
}
signatures["torch.nn.init.kaiming_uniform_"] = {
    "args": {
        "tensor": "tensor",
        "mode": "string",  # Could also be integer, representing the mode
        "nonlinearity": "string"  # Could also be 'leaky_relu' or a float
    },
    "kwargs": {
        "a": "float",
    },
    "inner": {},
}
signatures["torch.nn.init.normal_"] = {
    "args": {
        "tensor": "tensor",
        "mean": "float",
        "std": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.ones_"] = {
    "args": {
        "tensor": "tensor"  # Should it be Tensor or list of Tensors? Assuming tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.orthogonal_"] = {
    "args": {
        "m": "tensor"  # Should it be tensor_list as well?
    },
    "kwargs": {
        "gain": "float"  # or integer?
    },
    "inner": {}
}
signatures["torch.nn.init.sparse_"] = {
    "args": {
        "input": "tensor",
        "sparsity": "float",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.uniform_"] = {
    "args": {
        "tensor": "tensor",
        "a": "float",
        "b": "float"
    },
    "kwargs": {
        "dtype": "dtype" # could be torch.dtype
    },
    "inner": {},
}
signatures["torch.nn.init.xavier_normal_"] = {
    "args": {
        "tensor": "tensor"  # Should it be tensor or module? Assuming tensor.
    },
    "kwargs": {
        "gain": "float",
    },
    "inner": {},
}
signatures["torch.nn.init.xavier_uniform_"] = {
    "args": {
        "tensor": "tensor"  # could also be tensor_list
    },
    "kwargs": {
        "gain": "float"  # could be an integer too, but float seems more common
    },
    "inner": {},
}
signatures["torch.nn.init.zeros_"] = {
    "args": {
        "tensor": "tensor" # Could also be tensor_list, but the documentation focuses on tensor.
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.quantized.QFunctional"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {},
    "inner": {
        "args": {
            "scale": "tensor",  # Could be float also depending on the operation
            "zero_point": "tensor", # Could be integer also depending on the operation
        },
        "kwargs": {}
    }
}
signatures["torch.nn.utils.clip_grad_norm_"] = {
    "args": {
        "parameters": "tensor_list",
        "max_norm": "float",
        "norm_type": "integer" # Could also be float, depending on the value of p
    },
    "kwargs": {
        "eps": "float"
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_value_"] = {
    "args": {
        "parameters": "tensor_list",
        "clip_value": "float"
    },
    "kwargs": {
        "norm_type": "integer"  # Could be 2 or inf. I chose integer as it is a number.
    },
    "inner": {},
}
signatures["torch.nn.utils.get_total_norm"] = {
    "args": {
        "inputs": "tensor_list"  # Could also be a single tensor
    },
    "kwargs": {
        "norm_type": "float",  # could also be integer or string (inf, 1, 2)
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.utils.parameters_to_vector"] = {
    "args": {
        "params": "tensor_list",  # Could also be a single tensor
    },
    "kwargs": {
        "flatten_mode": "string", # Should it be 'start' or 'end'?
    },
    "inner": {},
}
signatures["torch.nn.utils.remove_weight_norm"] = {
    "args": {
        "module": "list" # Can be a module or a list of modules
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.utils.rnn.pack_padded_sequence"] = {
    "args": {
        "input": "tensor",
        "lengths": "tensor" # Could also be a list of integers
    },
    "kwargs": {
        "batch_first": "boolean",
        "padding_value": "float"
    },
    "inner": {},
}
signatures["torch.nn.utils.rnn.pack_sequence"] = {
    "args": {
        "input": "tensor_list",
        "lengths": "tensor"
    },
    "kwargs": {
        "batch_first": "boolean"  # Could also be None, but boolean is more representative.
    },
    "inner": {}
}
signatures["torch.nn.utils.rnn.pad_sequence"] = {
    "args": {
        "sequences": "tensor_list",
        "batch_first": "boolean",  # Could also be None
        "padding_value": "float"
    },
    "kwargs": {
        "padding_value": "float" #This is a duplicate argument.
    },
    "inner": {}
}
signatures["torch.nn.utils.vector_to_parameters"] = {
    "args": {
        "vector": "tensor",
        "modules": "list" # Could also be a tuple
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor",  # could be LongTensor
        "as_tuple": "boolean"
    },
    "inner": {}
}
signatures["torch.norm"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float",  # can be int, float, inf, -inf, 'fro', 'nuc'
        "dim": "tuple",  # can be int, tuple of ints, list of ints
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype",
    },
    "inner": {},
}
signatures["torch.normal_1"] = {
    "args": {
        "mean": "tensor",
        "std": "tensor"
    },
    "kwargs": {
        "generator": "torch.Generator",
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.normal_2"] = {
    "args": {
        "mean": "float",
        "std": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.normal_3"] = {
    "args": {
        "mean": "tensor",
        "std": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.normal_4"] = {
    "args": {
        "mean": "float",
        "std": "float",
        "size": "tuple"  # or list, but tuple seems more common for shapes
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.not_equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.numel"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.ones_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype", # Could be torch.dtype instead
        "layout": "string", # Could be torch.layout instead
        "device": "string", #device not included
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.outer"] = {
    "args": {
        "input": "tensor",
        "vec2": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.parse_schema"] = {
    "args": {
        "schema": "string"  # Could also be a dict, but string seems more appropriate here
    },
    "kwargs": {
        "strict": "boolean",
        "include_default_values": "boolean"
    },
    "inner": {},
}
signatures["torch.parse_type_comment"] = {
    "args": {
        "comment": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.pdist"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "p": "float",  # Could be integer as well
        "eps": "float"
    },
    "inner": {}
}
signatures["torch.permute"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"  # Could also be list of integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.pinverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "rcond": "float"  # could also be integer
    },
    "inner": {},
}
signatures["torch.polar"] = {
    "args": {
        "abs": "tensor",  # Could be float or double, represented as tensor
        "angle": "tensor"  # Must be same dtype as abs, represented as tensor
    },
    "kwargs": {
        "out": "tensor"  # If float32, complex64; if float64, complex128
    },
    "inner": {}
}
signatures["torch.polygamma"] = {
    "args": {
        "n": "integer",
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.positive"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.pow_1"] = {
    "args": {
        "input": "tensor",
        "exponent": "float"  # could also be tensor
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.pow_2"] = {
    "args": {
        "input": "tensor",
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.pow_3"] = {
    "args": {
        "self": "float",
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.prepare_multiprocessing_environment"] = {
    "args": {},
    "kwargs": {
        "workers": "integer" # Could also be None
    },
    "inner": {},
}
signatures["torch.prod_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.prod_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.promote_types"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list, but tensor is more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.put"] = {
    "args": {
        "tensor": "tensor",
        "indices": "tensor",
        "values": "tensor",
        "accumulate": "boolean"  # Could be a flag.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.qr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "some": "boolean",
        "out": "tuple" # Could also be list, but tuple seems more common
    },
    "inner": {}
}
signatures["torch.quantile"] = {
    "args": {
        "input": "tensor",
        "q": "float",  # Can also be a tensor
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "interpolation": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.quantize_per_channel"] = {
    "args": {
        "input": "tensor",
        "scale": "tensor",
        "zero_point": "tensor",
        "quantized_dtype": "dtype" # Should it be enum?
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.quantize_per_tensor"] = {
    "args": {
        "input": "tensor",
        "scale": "tensor",
        "zero_point": "tensor",
        "quantized_dtype": "dtype" # Could be string representing dtype like 'quint8'
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.rad2deg"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # potentially None, but tensor is the documented type
    },
    "inner": {}
}
signatures["torch.rand_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype", # Could be torch.dtype instead
        "layout": "string", # Could be torch.layout instead
        "device": "string", #removed device
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.randint_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "low": "integer", # Could also be tensor
        "high": "integer", # Could also be tensor
        "generator": "object", # Not specified but likely a generator object
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.randn_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype", # Could be torch.dtype instead
        "layout": "string", # Could be torch.layout instead
        "device": "string", # intentionally skipped
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.range"] = {
    "args": {
        "start": "float",
        "end": "float",
        "step": "float"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # could be torch.layout
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.ravel"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.reciprocal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.reciprocal_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor" # could be None
    },
    "inner": {}
}
signatures["torch.relu_"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list, but single tensor is more common
    },
    "kwargs": {
        "inplace": "boolean"  # This modifies the input tensor in-place
    },
    "inner": {}
}
signatures["torch.remainder"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.repeat_interleave_1"] = {
    "args": {
        "input": "tensor",
        "repeats": "tensor"  # Could also be integer, creating a separate signature
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.repeat_interleave_2"] = {
    "args": {
        "input": "tensor",
        "repeats": "integer"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.reshape"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple"  # Could also be a list of integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.roll"] = {
    "args": {
        "input": "tensor",
        "shifts": "tuple", # Can also be integer
        "dims": "tuple" # Can also be integer, or None
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.rot90"] = {
    "args": {
        "input": "tensor",
        "k": "integer",
    },
    "kwargs": {
        "dims": "list"  # or tuple
    },
    "inner": {},
}
signatures["torch.round"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "decimals": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.row_stack"] = {
    "args": {
        "tensors": "tensor_list"  # could also be a tuple of tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.rsqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.rsqrt_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.rsub"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float", # Could be a tensor too, but float seems more common.
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.scatter"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.scatter_add"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor"
    },
    "kwargs": {
        "reduce": "string"  # Should it be boolean?
    },
    "inner": {},
}
signatures["torch.scatter_reduce"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor"
    },
    "kwargs": {
        "reduce": "string"  # Could be an enum, using string as closest match
    },
    "inner": {},
}
signatures["torch.searchsorted"] = {
    "args": {
        "sorted_sequence": "tensor",
        "values": "tensor"  # or Scalar
    },
    "kwargs": {
        "out_int32": "boolean",
        "right": "boolean",
        "side": "string",
        "out": "tensor",
        "sorter": "tensor"  # LongTensor
    },
    "inner": {},
}
signatures["torch.select"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "integer"  # Could be tensor as well, but documentation specifies integer.
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.select_copy"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_anomaly_enabled"] = {
    "args": {
        "enabled": "boolean"  # Could also be integer (0 or 1)
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_cache_enabled"] = {
    "args": {
        "enabled": "boolean" # Could also be integer, but boolean seems more fitting based on documentation
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_cpu_dtype"] = {
    "args": {
        "dtype": "dtype"  # could also be a string representing the dtype
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_cpu_enabled"] = {
    "args": {},
    "kwargs": {
        "enabled": "boolean" # Could also be integer (0 or 1) but boolean seems more appropriate
    },
    "inner": {},
}
signatures["torch.set_autocast_enabled"] = {
    "args": {
        "enabled": "boolean"  # could also be integer (0 or 1)
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_ipu_dtype"] = {
    "args": {
        "dtype": "dtype"  # Could be a string representing the dtype as well
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_ipu_enabled"] = {
    "args": {},
    "kwargs": {
        "enabled": "boolean"
    },
    "inner": {}
}
signatures["torch.set_autocast_xla_enabled"] = {
    "args": {},
    "kwargs": {
        "enabled": "boolean"
    },
    "inner": {},
}
signatures["torch.set_default_device"] = {
    "args": {
        "device": "string" # could also be 'integer', representing the device index
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_default_dtype"] = {
    "args": {
        "dtype": "dtype"  # Could be torch.dtype instead of dtype
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_default_tensor_type"] = {
    "args": {
        "dtype": "dtype"  # Could be a string representing the dtype as well
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_deterministic_debug_mode"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.set_flush_denormal"] = {
    "args": {},
    "kwargs": {
        "flush": "boolean" # Should it be a boolean or integer? Assuming boolean as documented.
    },
    "inner": {},
}
signatures["torch.set_grad_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_num_interop_threads"] = {
    "args": {
        "num_threads": "integer"  # Could be a boolean as well, but integer seems more common.
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.set_num_threads"] = {
    "args": {
        "num_threads": "integer" # Could also be None, but integer is the main use case
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_warn_always"] = {
    "args": {},
    "kwargs": {
        "value": "boolean"  # Could also be string, depending on the intended behavior.
    },
    "inner": {},
}
signatures["torch.sgn"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.sigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.sigmoid_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.sign"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.signbit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.sin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.sin_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.sinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.slogdet"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.smm"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.sort"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",  # could also be None
        "descending": "boolean"
    },
    "kwargs": {
        "stable": "boolean",
        "out": "tuple"  # tuple of (Tensor, LongTensor)
    },
    "inner": {}
}
signatures["torch.sparse_bsr_tensor"] = {
    "args": {
        "indices": "tensor",
        "values": "tensor",
        "shape": "tuple",
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string" #device is skipped based on problem description
    },
    "inner": {},
}
signatures["torch.sparse_coo_tensor_1"] = {
    "args": {
        "indices": "tensor",
        "values": "tensor",
        "size": "tuple" # or list
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string", # skipped as requested
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_coo_tensor_2"] = {
    "args": {
        "indices": "tensor",
        "values": "tensor",
        "size": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string", # skipped as requested
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_csr_tensor"] = {
    "args": {
        "indices": "tensor",  # Could also be a list of tensors
        "values": "tensor",
        "size": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string"  # skipped as per instructions
    },
    "inner": {}
}
signatures["torch.special.bessel_j1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.bessel_y0"] = {
    "args": {
        "x": "tensor"  # Could also be tensor_list, but tensor seems more common
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.bessel_y1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tensor" # should it be tensor or None? choosing tensor
    },
    "inner": {},
}
signatures["torch.special.digamma"] = {
    "args": {
        "x": "tensor"  # could also be tensor_list
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.entr"] = {
    "args": {
        "x": "tensor"  # Could also be tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.erf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.erfc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.erfcx"] = {
    "args": {
        "input": "tensor" # Could also be a list of tensors, but sticking to tensor for now
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.erfinv"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.exp2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.expit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.expm1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.gammainc"] = {
    "args": {
        "input": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "out": "tensor" # should it be tensor or None? choosing tensor
    },
    "inner": {},
}
signatures["torch.special.gammaincc"] = {
    "args": {
        "x": "tensor",
        "a": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could also be None
    },
    "inner": {},
}
signatures["torch.special.gammaln"] = {
    "args": {
        "x": "tensor"  # could also be a list of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.hermite_polynomial_he"] = {
    "args": {
        "x": "tensor",
        "n": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.i0"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list, but tensor seems more common.
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.i0e"] = {
    "args": {
        "x": "tensor"  # Could be tensor_list as well
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.i1"] = {
    "args": {
        "x": "tensor" # Could also be tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.i1e"] = {
    "args": {
        "x": "tensor"  # could also be tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.log1p"] = {
    "args": {
        "x": "tensor"  # Could also be a list of tensors, but tensor seems most common.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.log_ndtr"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could also be None
    },
    "inner": {}
}
signatures["torch.special.log_softmax"] = {
    "args": {
        "input": "tensor"  # Could also be a list of tensors
    },
    "kwargs": {
        "dim": "integer",  # Specifies the dimension along which softmax is computed
        "dtype": "dtype"  # The desired data type of returned tensors.
    },
    "inner": {},
}
signatures["torch.special.logit"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "eps": "float"  # Could also be a tensor, but float is more general
    },
    "inner": {},
}
signatures["torch.special.logsumexp"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "dim": "integer",  # Could be None
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.special.modified_bessel_i0"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.modified_bessel_i1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.modified_bessel_k0"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.multigammaln"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",  # Could also be integer
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.special.ndtr"] = {
    "args": {
        "x": "tensor"  # could also be a list of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.ndtri"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.polygamma"] = {
    "args": {
        "n": "integer",  # Could also be float, but integer seems more common for the order
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.psi"] = {
    "args": {
        "x": "tensor"  # Could also be tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.round"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "mode": "string",  # could be 'trunc', 'floor', 'ceil'
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.sinc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.softmax"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list
    },
    "kwargs": {
        "dim": "integer", # Could be None
        "dtype": "dtype" # Maybe tensor?
    },
    "inner": {},
}
signatures["torch.special.spherical_bessel_j0"] = {
    "args": {
        "x": "tensor"  # could also be tensor_list
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.xlog1py"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "threshold": "float" # Could also be an integer
    },
    "inner": {},
}
signatures["torch.special.xlogy"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.zeta"] = {
    "args": {
        "x": "tensor"  # could also be a list of tensors
    },
    "kwargs": {
        "q": "float"  # Could potentially be a tensor, but float seems more common
    },
    "inner": {},
}
signatures["torch.split"] = {
    "args": {
        "tensor": "tensor",
        "split_size_or_sections": "list",  # could also be integer
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.spmm"] = {
    "args": {
        "mat1": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "sparse_col": "boolean", # could be integer for the column index
    },
    "inner": {},
}
signatures["torch.sqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.sqrt_"] = {
    "args": {
        "input": "tensor"  # could also be a list of tensors
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.square"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well, but tensor is more likely
    },
    "inner": {},
}
signatures["torch.square_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Should it be tensor or optional tensor?
    },
    "inner": {},
}
signatures["torch.squeeze"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"  # or tuple
    },
    "inner": {}
}
signatures["torch.sspaddmm"] = {
    "args": {
        "input": "tensor",
        "mat1": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "beta": "float",  # could be number, defaulting to float
        "alpha": "float",  # could be number, defaulting to float
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.stack"] = {
    "args": {
        "tensors": "tensor_list",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"  # Could be None, but tensor is more appropriate
    },
    "inner": {}
}
signatures["torch.std"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # could also be a tuple of integers
    },
    "kwargs": {
        "correction": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.std_mean"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer", # Could be a tuple of integers
        "keepdim": "boolean",
        "dtype": "dtype" # potentially None
    },
    "inner": {},
}
signatures["torch.stft"] = {
    "args": {
        "input": "tensor",
        "n_fft": "integer",
        "hop_length": "integer",
        "win_length": "integer",
    },
    "kwargs": {
        "window": "tensor",  # could be None
        "center": "boolean",
        "pad_mode": "string",
        "normalized": "boolean",
        "onesided": "boolean",
        "return_complex": "boolean",
        "align_to_window": "integer", # doc isn't clear on type, assuming integer
    },
    "inner": {},
}
signatures["torch.sub"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Could also be number
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.subtract"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor" # Could be tensor or None
    },
    "inner": {},
}
signatures["torch.sum_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.sum_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # or tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.svd"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "some": "boolean",  # Could also be interpreted as integer, but documentation emphasizes boolean control
        "compute_uv": "boolean",
        "out": "tuple"  # Expecting a tuple of tensors
    },
    "inner": {},
}
signatures["torch.swapaxes"] = {
    "args": {
        "input": "tensor",
        "axis0": "integer",
        "axis1": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.swapdims"] = {
    "args": {
        "input": "tensor",
        "dim0": "integer",
        "dim1": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sym_float"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sym_fresh_size"] = {
    "args": {
        "tensor": "tensor" # Could also be a list of tensors, but single tensor is more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sym_int"] = {
    "args": {
        "x": "tensor"  # Could also be float or integer, but tensor seems most general.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.t"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.take"] = {
    "args": {
        "input": "tensor",
        "index": "tensor"  # should be LongTensor, but tensor is closest
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.tan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.tan_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.tanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.tensordot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "dims": "integer"  # or Tuple[List[int], List[int]] or List[List[int]]
    },
    "inner": {}
}
signatures["torch.threshold"] = {
    "args": {
        "input": "tensor",
        "threshold": "float",
    },
    "kwargs": {
        "value": "tensor",  # Could be a number as well. Using tensor as it's more general.
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.tile"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"  # Could also be integer, but tuple seems more general based on documentation
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.topk"] = {
    "args": {
        "input": "tensor",
        "k": "integer",
        "dim": "integer" # Could also be None
    },
    "kwargs": {
        "largest": "boolean",
        "sorted": "boolean",
        "out": "tuple" # tuple of (Tensor, LongTensor)
    },
    "inner": {},
}
signatures["torch.transpose"] = {
    "args": {
        "input": "tensor",
        "dim0": "integer",
        "dim1": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.trapz"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "dim": "integer"  # could also be None
    },
    "inner": {},
}
signatures["torch.triangular_solve"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "upper": "boolean",  # Could be boolean or None
        "unit": "boolean"   # Could be boolean or None
    },
    "inner": {}
}
signatures["torch.tril"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"  # could also be float, but integer is more common
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.tril_indices"] = {
    "args": {
        "rows": "integer",
        "cols": "integer"
    },
    "kwargs": {
        "offset": "integer",  # Could also be a tensor, but integer seems more common
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.triu"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"  # could also be float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.triu_indices"] = {
    "args": {
        "rows": "integer",
        "cols": "integer",
        "offset": "integer"  # Could be float as well, defaulting to int
    },
    "kwargs": {
        "device": "string" # Removed device
    },
    "inner": {},
}
signatures["torch.true_divide"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rounding_mode": "string"  # Could be None, but string seems closest
    },
    "inner": {}
}
signatures["torch.trunc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.trunc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.typename"] = {
    "args": {
        "obj": "tensor" # could also be other types like module, etc. but documentation says tensor
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unbind"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unique"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "sorted": "boolean",  # Could also be None, but boolean is closer.
        "return_inverse": "boolean",
        "return_counts": "boolean",
        "dim": "integer"  # Could be None, but integer is closer.
    },
    "inner": {}
}
signatures["torch.unique_consecutive"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "return_inverse": "boolean",
        "return_counts": "boolean"
    },
    "inner": {}
}
signatures["torch.unravel_index"] = {
    "args": {
        "index": "tensor",
        "dims": "tuple" # Could also be a list of integers, but tuple seems more specific here
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unsafe_split_with_sizes"] = {
    "args": {
        "input": "tensor",
        "sizes": "list"  # Could also be a tuple of integers
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.unsqueeze"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # could also be a tuple, but documentation only shows integer
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.vander"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "N": "integer",  # Could be None, but integer is the primary type
        "increasing": "boolean"
    },
    "inner": {}
}
signatures["torch.var"] = {
    "args": {
        "input": "tensor",
        "dim": "integer" # or tuple of integers, or None
    },
    "kwargs": {
        "correction": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.var_mean"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "keepdim": "boolean", # could be integer
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.vdot"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # potentially tensor or None
    },
    "inner": {}
}
signatures["torch.view_as_complex"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.view_as_complex_copy"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.view_as_real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.vitals_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.vsplit"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"  # could also be integer
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.vstack"] = {
    "args": {
        "tensors": "tensor_list"  # could also be a single tensor
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.where"] = {
    "args": {
        "condition": "tensor",
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None
    },
    "inner": {}
}
signatures["torch.where_1"] = {
    "args": {
        "condition": "tensor",
        "input": "float",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None
    },
    "inner": {}
}
signatures["torch.where_2"] = {
    "args": {
        "condition": "tensor",
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None
    },
    "inner": {}
}
signatures["torch.where_3"] = {
    "args": {
        "condition": "tensor",
        "input": "float",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None
    },
    "inner": {}
}
signatures["torch.zeros"] = {
    "args": {
        "size": "list" # could also be a tuple or multiple integers
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",
        "device": "string", # intentionally omitted as per instructions
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.zeros_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype", # Could be torch.dtype
        "layout": "string", # Could be torch.layout
        "device": "string", # Skipping device as per instructions
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.asinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.gcd"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.gcd__1"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.gcd__2"] = {
    "args": {
        "input": "integer",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.gcd__3"] = {
    "args": {
        "input": "integer",
        "other": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list, but tensor seems more common.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.trunc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.asinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list, but tensor seems more common.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.trunc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.asinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list, but tensor seems more common.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.trunc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.asinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list, but tensor seems more common.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.trunc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
