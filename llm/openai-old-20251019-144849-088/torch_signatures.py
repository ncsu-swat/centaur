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
        "out": "tensor"  # could be None as well, but tensor is more specific
    },
    "inner": {},
}
signatures["torch.abs_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
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
    "inner": {}
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
        "value": "float",  # could also be integer
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
        "value": "float",  # could be integer as well, but documentation says float/double for FloatTensor/DoubleTensor
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
    "inner": {},
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
        "dim": "integer",  # Can also be a tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor",
    },
    "inner": {},
}
signatures["torch.amin"] = {
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
signatures["torch.aminmax"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",  # could be None
        "keepdim": "boolean",
        "out": "tuple"  # Expects a tuple of two tensors
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
        "layout": "string",  # Assuming layout is a string representing the layout type
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.arccos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor" # potentially tensor or None
    },
    "inner": {}
}
signatures["torch.arccos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
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
        "out": "tensor"  # could be None as well
    },
    "inner": {},
}
signatures["torch.arcsin_"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.arcsinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well
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
        "out": "tensor"  # Could also be None
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
        "out": "tensor"  # could be None as well
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
    "inner": {}
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
        "keepdim": "boolean" # could also be None, but boolean seems more fitting here
    },
    "inner": {}
}
signatures["torch.argmin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # could also be None
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.argsort"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
    },
    "kwargs": {
        "descending": "boolean",
        "stable": "boolean"  # Could also be None, but boolean is most accurate based on doc
    },
    "inner": {},
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
        "obj": "tensor"  # Could be a NumPy array, DLPack capsule, scalar, or sequence of scalars
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "requires_grad": "boolean"
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
        "out": "tensor"  # could be None as well, but tensor is more representative.
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
        "out": "tensor"  # potentially None, but tensor is the documented type
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
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.atanh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could be None too, but tensor is the more common case
    },
    "inner": {}
}
signatures["torch.atleast_1d"] = {
    "args": {
        "input": "tensor"  # could also be tensor_list
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.atleast_2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.atleast_3d"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.autocast"] = {
    "args": {
        "enabled": "boolean"  # Could also be integer (0 or 1)
    },
    "kwargs": {
        "dtype": "dtype", # Might be better as torch.dtype
        "cache_enabled": "boolean"
    },
    "inner": {}
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
        "length": "integer",  # Could also be float
        "periodic": "boolean",
        "beta": "float"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string"
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
        "weights": "tensor",  # Could be None, but documented as tensor
        "minlength": "integer",
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
        "shift": "tensor"  # Could also be integer
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
        "out": "tensor"  # Should it be tensor or optional tensor? Assuming tensor for now
    },
    "inner": {},
}
signatures["torch.bitwise_right_shift"] = {
    "args": {
        "input": "tensor",
        "shift": "integer"  # Could be tensor as well, but integer seems more common
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
        "length": "integer", # Could also be a tuple
        "periodic": "boolean",
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # Could be tensor
    },
    "inner": {},
}
signatures["torch.block_diag"] = {
    "args": {
        "matrices": "tensor_list"  # could also be a single tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.bmm"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype",  # could also be None
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.broadcast_shapes"] = {
    "args": {
        "shapes": "list" # Could also be a tuple
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.broadcast_tensors"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.broadcast_to_1"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.broadcast_to_2"] = {
    "args": {
        "input": "tensor",
        "shape": "list"
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
        "dim": "integer"  # Could also be a tuple, but documentation doesn't explicitly state
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
        "out": "tensor"  # could also be None
    },
    "inner": {},
}
signatures["torch.ceil_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.celu"] = {
    "args": {
        "input": "tensor",
        "alpha": "float"  # Could be a tensor as well, but float seems more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.chain_matmul"] = {
    "args": {
        "tensors": "tensor_list" # Could also be a tuple of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.channel_shuffle"] = {
    "args": {
        "input": "tensor",
        "groups": "integer"  # could also be a tuple, but integer seems more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cholesky"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "upper": "boolean",
        "out": "tensor" # could be None
    },
    "inner": {},
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
    "inner": {},
}
signatures["torch.chunk"] = {
    "args": {
        "input": "tensor",
        "chunks": "integer",
        "dim": "integer"  # could also be a tuple, but the documentation only refers to integer
    },
    "kwargs": {},
    "inner": {}
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
        "max": "float" # could also be tensor
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
    "kwargs": {},
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
        "real": "tensor",  # could be half, float, or double, but all are tensors
        "imag": "tensor"  # Must be the same dtype as real, so also a tensor
    },
    "kwargs": {
        "out": "tensor"  # out is a tensor, the documentation specifies its dtype depending on input
    },
    "inner": {}
}
signatures["torch.concat"] = {
    "args": {
        "tensors": "tensor_list",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"  # Should it be tensor or None? Assuming tensor.
    },
    "inner": {},
}
signatures["torch.concatenate"] = {
    "args": {
        "tensors": "tensor_list",
        "axis": "integer"
    },
    "kwargs": {
        "out": "tensor"  # Could be None, but tensor is a better fit
    },
    "inner": {}
}
signatures["torch.conj"] = {
    "args": {
        "input": "tensor"  # Could potentially be a list of tensors, but documentation suggests single tensor
    },
    "kwargs": {},
    "inner": {},
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
        "input": "tensor"  # Could be scalar or 1D vector as well
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.cos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.cosh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor" # could be None as well
    },
    "inner": {}
}
signatures["torch.count_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer", # Could be a tuple of integers as well
        "keepdim": "boolean",
        "dtype": "dtype" # It can also be None
    },
    "inner": {},
}
signatures["torch.cross"] = {
    "args": {
        "input": "tensor",
        "other": "tensor",
        "dim": "integer"  # could also be None
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.crow_indices_copy"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could also be None
    },
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
        "out": "tuple"  # Should it be tensor? Tuple of tensors?
    },
    "inner": {}
}
signatures["torch.cummin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tuple"  # could also be None
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
        "out": "tensor"  # could also be None, but tensor is more specific
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
        "input": "tensor"  # Could be tensor_list as well, but documentation only shows a tensor
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.device"] = {
    "args": {
        "type": "string",  # could also be integer, but string seems more common
        "device": "integer" # could be string, but integer is more common based on documentation
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.diag"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"
    },
    "kwargs": {
        "out": "tensor"  # Could be None, but tensor is more general
    },
    "inner": {}
}
signatures["torch.diag_embed"] = {
    "args": {
        "input": "tensor",
        "offset": "integer",
        "dim1": "integer",
        "dim2": "integer"
    },
    "kwargs": {
        "out": "tensor" # Could also be None
    },
    "inner": {}
}
signatures["torch.diagflat"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "offset": "integer"  # Could be int, but integer is more strict
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
        "prepend": "tensor", # could also be None
        "append": "tensor" # could also be None
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
    "inner": {}
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
        "other": "tensor"  # could also be a number (float or integer)
    },
    "kwargs": {
        "rounding_mode": "string", # could be None
        "out": "tensor"  # could be None
    },
    "inner": {}
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
    "inner": {}
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
        "operands": "tensor_list"  # could also be a single tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.empty"] = {
    "args": {
        "size": "list" # or tuple, or a variable number of integers
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # torch.layout
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "string" # torch.memory_format
    },
    "inner": {}
}
signatures["torch.empty_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # Could be torch.layout, but string is close enough
        "device": "string", # Skipping device as requested
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.empty_strided"] = {
    "args": {
        "size": "tuple", # could also be integer or list
        "stride": "tuple", # could also be list
        "dtype": "dtype"
    },
    "kwargs": {
        "layout": "string", # could be torch.layout
        "device": "string", # skipping device
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.eq"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # or float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
        "input": "tensor"  # Could be tensor_list as well, but documentation doesn't mention that
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
        "out": "tensor"  # Could be tensor or None
    },
    "inner": {},
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
        "out": "tensor"  # could be None, but tensor is more specific
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
        "sizes": "tuple" # or list, but tuple seems more appropriate
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
        "m": "integer"  # Could be optional, but documentation states default is n
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # torch.layout
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.fake_quantize_per_channel_affine"] = {
    "args": {
        "input": "tensor",
        "fake_quant_params": "tuple" # Could also be a list, but tuple is more common for fixed-size parameters
    },
    "kwargs": {
        "quantize_dtype": "dtype", # Could be torch.dtype
        "scale": "tensor",
        "zero_point": "tensor"
    },
    "inner": {}
}
signatures["torch.fake_quantize_per_tensor_affine"] = {
    "args": {
        "input": "tensor",
        "quantize_scale": "float",
        "quantize_zero_point": "integer"
    },
    "kwargs": {
        "fake_quant_mode": "string", # Could also be an enum
        "observer": "object" # Assuming this is an observer object
    },
    "inner": {}
}
signatures["torch.fft.fft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",  # Could be None
        "dim": "integer"
    },
    "kwargs": {
        "norm": "string",
        "out": "tensor"  # Could be None
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
        "d": "float"  # Could be integer as well, but float is more general
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # torch.layout is a string
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.fft.fftn"] = {
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
signatures["torch.fft.fftshift"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer" # or tuple[int]
    },
    "inner": {}
}
signatures["torch.fft.hfft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",  # Could be None
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
        "n": "integer",  # Could be None
        "dim": "integer"
    },
    "kwargs": {
        "norm": "string",
        "out": "tensor"  # Could be None
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
        "dim": "integer"  # Or tuple[int], but sticking with integer for simplicity
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
        "dim": "tuple",  # can be None
        "norm": "string",  # can be None
        "out": "tensor"  # can be None
    },
    "inner": {},
}
signatures["torch.fft.irfftn"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "s": "tuple",  # can be None
        "dim": "tuple",  # can be None
        "norm": "string",  # can be None
        "out": "tensor",  # can be None
    },
    "inner": {},
}
signatures["torch.fft.rfft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",  # Could be None
        "dim": "integer"
    },
    "kwargs": {
        "norm": "string",  # Could be "forward", "backward", or "ortho"
        "out": "tensor"  # Could be None
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
        "s": "tuple",  # Could be a list as well
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.finfo"] = {
    "args": {
        "dtype": "dtype"  # Could be a string representing the dtype
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.fix"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None, but tensor is more specific.
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
        "other": "tensor" # could also be integer
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor_divide_1"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
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
        "out": "tensor"  # could be None, but tensor is more specific
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
    "inner": {},
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
        "dtype": "dtype",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.from_numpy"] = {
    "args": {
        "input": "tensor" # Could also be list of tensors
    },
    "kwargs": {
        "dtype": "dtype" #Could also be torch.dtype
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
        "dtype": "dtype",
        "layout": "string",
        "device": "string", # skipped
        "requires_grad": "boolean",
        "memory_format": "string"
    },
    "inner": {},
}
signatures["torch.gather"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor"  # Could also be LongTensor, but "tensor" is more general
    },
    "kwargs": {
        "sparse_grad": "boolean",
        "out": "tensor"
    },
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
    "inner": {}
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
        "cache_dir": "string", # Could be None
        "hash": "string", # Could be None
        "progress": "boolean"
    },
    "inner": {},
}
signatures["torch.get_num_threads"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.get_rng_state"] = {
    "args": {},
    "kwargs": {
        "device": "integer" # Could also be string, but integer is more specific based on documentation
    },
    "inner": {},
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
signatures["torch.greater_equal_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.greater_equal_2"] = {
    "args": {
        "input": "tensor",
        "other": "float" # could also be integer
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.greater_equal_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {},
    "inner": {}
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
        "length": "integer",  # could also be tensor
        "periodic": "boolean",
        "alpha": "float",
        "beta": "float",
        "window_type": "string"
    },
    "kwargs": {
        "dtype": "dtype",  # might be optional
        "layout": "string" # might be optional
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
        "layout": "string",
        "device": "string",  # Not including device
        "requires_grad": "boolean"
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
        "out": "tensor" # could also be None
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
    "inner": {}
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
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
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
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.i0"] = {
    "args": {
        "input": "tensor"  # Could be tensor_list as well, but documentation doesn't explicitly state it
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
        "out": "tensor" # Could also be None
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
        "input": "tensor"  # It should be a complex tensor, but "tensor" is the best option here.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.index_add_1"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "source": "tensor"
    },
    "kwargs": {
        "reduction": "string" # Could be enum, defaulting to 'add'
    },
    "inner": {}
}
signatures["torch.index_add_2"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "source": "tensor"
    },
    "kwargs": {
        "reduction": "string" # Could be enum, defaulting to 'add'
    },
    "inner": {}
}
signatures["torch.index_copy"] = {
    "args": {
        "input": "tensor",
        "index": "tensor",
        "source": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could be None
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
    "inner": {}
}
signatures["torch.is_autocast_cache_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_autocast_cpu_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_autocast_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_autocast_ipu_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_autocast_xla_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
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
        "dim": "integer", # Could also be None
        "out": "tensor",
        "invert": "boolean"
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
        "input": "tensor"  # Could also be tensor_list, but tensor seems more common.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_tensor"] = {
    "args": {
        "obj": "tensor" # Could also be 'list' or other Python objects
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
    "inner": {},
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
        "module": "tensor", # Could also be a string representing the module name
    },
    "kwargs": {},
    "inner": {
        "args": {
            "inputs": "tensor_list" # Shape: (N, ...) where N is the number of inputs.
        },
        "kwargs": {}
    }
}
signatures["torch.jit.Error"] = {
    "args": {
        "message": "string" # Could also be tensor, but string seems more appropriate.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.ScriptWarning"] = {
    "args": {
        "message": "string", # Could also be tensor, but string seems more common
    },
    "kwargs": {
        "user_defined_type": "string", # Seems like a string representing the custom type
        "smtype": "string", # Should be string
    },
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
        "drop": "boolean" # Could also be None, but boolean seems more appropriate
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
    "inner": {}
}
signatures["torch.jit.isinstance"] = {
    "args": {
        "obj": "tensor", # Could be Any, but assuming tensor for now.
        "target_type": "list" # Can be a type like List[str], Dict[str, List[torch.Tensor]], etc.
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.optimized_execution"] = {
    "args": {
        "module": "tensor"  # Could also be a list of tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.script_if_tracing"] = {
    "args": {
        "input": "tensor" # Could also be a list of tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.set_fusion_strategy"] = {
    "args": {
        "module": "tensor" # Could also be a list of tensors, but tensor is the primary use case
    },
    "kwargs": {
        "strategy": "string"
    },
    "inner": {},
}
signatures["torch.jit.set_module"] = {
    "args": {
        "name": "string",
        "module": "tensor"  # Could also be a callable, but tensor seems most accurate based on documentation
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.strict_fusion"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.wait"] = {
    "args": {
        "future": "tensor" # Should be torch.jit.Future[T], but using tensor as a close match.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.kaiser_window"] = {
    "args": {
        "window_length": "integer",
        "periodic": "boolean",
        "beta": "float"
    },
    "kwargs": {
        "dtype": "dtype" # Could be torch.dtype
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
        "dim": "integer",  # Could be None
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"  # potentially a tuple of tensors
    },
    "inner": {},
}
signatures["torch.lcm"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.ldexp"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # could also be integer
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.ldexp_"] = {
    "args": {
        "input": "tensor",
        "exponent": "tensor"
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
        "weight": "float"  # Could also be "tensor"
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
    "inner": {}
}
signatures["torch.lgamma"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {},
}
signatures["torch.linalg.cholesky"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.linalg.cholesky_ex"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "upper": "boolean" # Could also be a string, but boolean is closer
    },
    "inner": {}
}
signatures["torch.linalg.det"] = {
    "args": {
        "A": "tensor"  # Shape is (*, n, n), so tensor is the correct type.
    },
    "kwargs": {
        "out": "tensor"  # Optional output tensor.
    },
    "inner": {},
}
signatures["torch.linalg.eig"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tuple"  # could be None
    },
    "inner": {}
}
signatures["torch.linalg.eigh"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "UPLO": "string",
        "out": "tuple"  # Could be None
    },
    "inner": {}
}
signatures["torch.linalg.eigvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor" # could be None, but treating as tensor for simplicity
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
        "A": "tensor",  # Could also be tensor_list
        "u": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.inv"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor" # Could be None, but tensor is more general
    },
    "inner": {},
}
signatures["torch.linalg.lstsq"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "rcond": "float", # Could also be None
        "driver": "string" # Could be None
    },
    "inner": {}
}
signatures["torch.linalg.lu"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "pivot": "boolean",
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.linalg.matrix_norm"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ord": "string", # Could also be integer or float, but string is more general
        "dim": "integer",
        "keepdim": "boolean",
        "eps": "float"
    },
    "inner": {},
}
signatures["torch.linalg.matrix_power"] = {
    "args": {
        "input": "tensor",
        "n": "integer"  # Could be float, but integer seems more common for power
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
        "abs_tol": "float",
    },
    "inner": {},
}
signatures["torch.linalg.multi_dot"] = {
    "args": {
        "tensors": "tensor_list" # Could also be a tuple of tensors
    },
    "kwargs": {},
    "inner": {},
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
        "mode": "string",
        "out": "tuple"  # potentially a tuple of tensors
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
    "inner": {}
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
        "upper": "boolean",
        "lower": "boolean",
        "hermitian": "boolean",
        "check_finite": "boolean"
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
        "driver": "string",  # could be None
        "out": "tuple"  # tuple of tensors
    },
    "inner": {}
}
signatures["torch.linalg.svdvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "driver": "string",  # could also be None
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.linalg.tensorinv"] = {
    "args": {
        "A": "tensor",
        "ind": "integer"
    },
    "kwargs": {
        "out": "tensor" # Could be None, but defaulting to tensor
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
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.log"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {},
}
signatures["torch.log10"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.log1p"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.log2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well
    },
    "inner": {},
}
signatures["torch.log_"] = {
    "args": {
        "input": "tensor",
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
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logaddexp2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logcumsumexp"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"  # potentially tensor or None
    },
    "inner": {}
}
signatures["torch.logdet"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.logical_and"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {}
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
        "out": "tensor"  # could be None
    },
    "inner": {},
}
signatures["torch.logical_xor"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None
    },
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "eps": "float", # Could also be None
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list, but tensor seems more common.
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.logspace"] = {
    "args": {
        "start": "float",  # Could also be tensor
        "end": "float",  # Could also be tensor
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
        "dim": "integer",  # Or tuple of integers
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
        "pivot": "tensor"
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
        "out": "tensor"  # could also be None
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
        "out": "tuple"  # Should it be tuple?
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
        "out": "tensor"  # could also be None, but tensor is more specific
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
        "dim": "integer"  # Could also be None
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"  # (Tensor, Tensor) - First Tensor, Second Tensor (dtype long)
    },
    "inner": {}
}
signatures["torch.meshgrid"] = {
    "args": {
        "tensors": "tensor_list"  # could also be a list of scalars
    },
    "kwargs": {
        "indexing": "string"  # Could be 'xy' or 'ij', defaulting to 'ij'
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
        "out": "tuple" # Should be tuple of tensors (min, indices)
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
        "out": "tensor"  # could also be None, but tensor is more specific
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
        "out": "tensor"
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
        "source": "integer", # Could also be tuple
        "destination": "integer" # Could also be tuple
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.movedim"] = {
    "args": {
        "input": "tensor",
        "source": "tuple", # Can also be an integer
        "destination": "tuple" # Can also be an integer
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.msort"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None
    },
    "inner": {}
}
signatures["torch.mul"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # or float/integer
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.multiply"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor" # could also be None
    },
    "inner": {},
}
signatures["torch.mv"] = {
    "args": {
        "input": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
    "inner": {}
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
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple" # Could be (tensor, tensor)
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
        "dim": "integer"  # or tuple of integers
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
        "start": "integer",  # could also be tensor
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
        "group": "integer" # Could also be a tuple, but let's start with integer
    },
    "kwargs": {},
    "inner": {}
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
        "out": "tensor" # could also be None
    },
    "inner": {},
}
signatures["torch.negative_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float" # Could also be tensor, but float seems more common.
    },
    "inner": {}
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
        "output_size": "union[integer, tuple]" # Could be int or tuple of ints
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # Shape: (N,C,Lin) or (C,Lin)
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveAvgPool2d"] = {
    "args": {
        "output_size": "tuple"  # could also be int or None
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
        "output_size": "tuple"  # Could also be integer
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
        "cutoffs": "list" # could also be tuple
    },
    "kwargs": {
        "div_value": "float",
        "head_bias": "boolean",
        "dtype": "dtype" # assuming it's a dtype
    },
    "inner": {
        "args": {
            "input_": "tensor",
            "target_": "tensor" # could also be integer
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
        "output_size": "union[integer, tuple]" # Could be int or tuple of ints
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor" #Based on Shape: Input
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
        "p": "float"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.AvgPool1d"] = {
    "args": {
        "kernel_size": "union[int, tuple[int]]",  # could be int or tuple
        "stride": "union[int, tuple[int]]",  # could be int or tuple, defaults to kernel_size
        "padding": "union[int, tuple[int]]",  # could be int or tuple
    },
    "kwargs": {
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
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
        "kernel_size": "tuple",  # Could be int or tuple
        "stride": "tuple",  # Could be int or tuple, defaults to kernel_size
        "padding": "tuple",  # Could be int or tuple, defaults to 0
    },
    "kwargs": {
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer",  # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool3d"] = {
    "args": {
        "kernel_size": "tuple",  # could also be integer
        "stride": "tuple",  # could also be integer
        "padding": "tuple",  # could also be integer
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # optional
    },
    "kwargs": {},
    "inner": {},
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
        "weight": "tensor",  # Should it be tensor_list? No, it says "Tensor of size nbatch"
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
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
        "dtype": "dtype" # Added dtype
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
    "inner": {},
}
signatures["torch.nn.BatchNorm3d"] = {
    "args": {
        "num_features": "integer",
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Could also be None
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype" # added dtype
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
        "alpha": "float", # could be number, but float is more specific
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
        "blank": "integer",  # could also be float
    },
    "kwargs": {
        "reduction": "string",
        "zero_infinity": "boolean"
    },
    "inner": {
        "args": {
            "log_probs": "tensor",
            "targets": "tensor",
            "input_lengths": "tensor_list", # or tuple, or tensor
            "target_lengths": "tensor_list" # or tuple, or tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.CircularPad1d"] = {
    "args": {
        "padding": "tuple" # could also be integer
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad1d"] = {
    "args": {
        "padding": "tuple" # Could also be integer
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
signatures["torch.nn.ConstantPad2d"] = {
    "args": {
        "padding": "tuple" # Could also be integer
    },
    "kwargs": {
        "value": "float"
    },
    "inner": {},
}
signatures["torch.nn.ConstantPad3d"] = {
    "args": {
        "padding": "tuple"  # can also be an integer
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
        "padding": "integer",  # or tuple or string
        "dilation": "integer",  # or tuple
        "groups": "integer",
        "bias": "boolean",
    },
    "kwargs": {
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
        "padding": "string",  # can be int, tuple or string
        "dilation": "tuple",  # can be int or tuple
        "groups": "integer",
        "bias": "boolean"
    },
    "kwargs": {
        "padding_mode": "string",
        "dtype": "dtype" # added dtype
    },
    "inner": {},
}
signatures["torch.nn.Conv3d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple",  # or integer
        "stride": "tuple",  # or integer
        "padding": "tuple",  # or integer or string
        "dilation": "tuple",  # or integer
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
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "output_size": "list"  # or None
        },
        "kwargs": {}
    }
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
    "inner": {}
}
signatures["torch.nn.CosineSimilarity"] = {
    "args": {},
    "kwargs": {
        "dim": "integer", # could be int or None
        "eps": "float"
    },
    "inner": {
        "args": {
            "x1": "tensor",
            "x2": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.CrossEntropyLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"  # Could also be a list of tensors
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
    "args": {
        "p": "float" # Could also be interpreted as a tensor, but documentation clearly states float
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor" # Based on the example and documentation
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Dropout2d"] = {
    "args": {
        "p": "float" # Could also be interpreted as a probability value.
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
signatures["torch.nn.Dropout3d"] = {
    "args": {
        "p": "float" # could also be integer, but float is more common
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor" # Input shape: (N,C,D,H,W) or (C,D,H,W)
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ELU"] = {
    "args": {},
    "kwargs": {
        "alpha": "float", # could be number
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
        "norm_type": "float", # could be integer as well
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.EmbeddingBag"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer",
        "max_norm": "float",  # Could be None
        "norm_type": "float",  # Default 2.0
        "scale_grad_by_freq": "boolean",  # Default False
        "mode": "string",  # Default: "mean"
        "sparse": "boolean",  # Default False
        "include_last_offset": "boolean",  # Default False
        "padding_idx": "integer",  # Could be None
        "dtype": "dtype" # Added dtype
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "offsets": "tensor",
            "per_sample_weights": "tensor"  # Could be None
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FeatureAlphaDropout"] = {
    "args": {
        "p": "float" # Should it be tensor? No, it's probability, so float.
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor" # From Shape: Input: (N,C,D,H,W) or (C,D,H,W)
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
        "end_dim": "integer"   # Could also be None
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
    "inner": {},
}
signatures["torch.nn.FractionalMaxPool2d"] = {
    "args": {
        "kernel_size": "union[integer, tuple]",  # Could be int or tuple
        "output_size": "union[integer, tuple]",  # Could be int or tuple
        "output_ratio": "union[float, tuple]",  # Could be float or tuple
        "return_indices": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.FractionalMaxPool3d"] = {
    "args": {
        "kernel_size": "union[int, tuple]",  # Could be int or tuple of ints
        "output_size": "union[int, tuple]",  # Could be int or tuple of ints
        "output_ratio": "union[float, tuple]",  # Could be float or tuple of floats
        "return_indices": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.GELU"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "approximate": "string" # Could also be an Enum, but string is more general.
    },
    "inner": {}
}
signatures["torch.nn.GLU"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer" # Could also be None, but integer is more precise
    },
    "inner": {}
}
signatures["torch.nn.GRU"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
        "num_layers": "integer",
        "input": "tensor",  # Could also be packed sequence
        "h_0": "tensor"  # Could also be a tensor_list
    },
    "kwargs": {
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.GRUCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
    },
    "kwargs": {
        "bias": "boolean", # Could also be None
        "dtype": "dtype" # Not explicitly stated but common for nn.Module
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hidden": "tensor" # Could be None
        },
        "kwargs": {}
    }
}
signatures["torch.nn.GaussianNLLLoss"] = {
    "args": {},
    "kwargs": {
        "full": "boolean",  # could be boolean or None
        "eps": "float",
        "reduction": "string"  # could be 'none', 'mean', 'sum'
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor",
            "var": "tensor" # or scalar
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
        "eps": "float",  # Should it be a dtype instead?
        "affine": "boolean",
        "dtype": "dtype"  # Could also be None
    },
    "inner": {},
}
signatures["torch.nn.Hardshrink"] = {
    "args": {
        "lambd": "float" # could also be a tensor, but documentation suggests a float
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.Hardsigmoid"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list, but documentation says any number of dimensions so tensor is more general
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.Hardswish"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.Hardtanh"] = {
    "args": {
        "min_val": "float",
        "max_val": "float",
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.HingeEmbeddingLoss"] = {
    "args": {
        "margin": "float"
    },
    "kwargs": {
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
    "inner": {},
}
signatures["torch.nn.InstanceNorm1d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.InstanceNorm2d"] = {
    "args": {
        "num_features": "integer",
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "kwargs": {
        "device": "dtype", # Should have been dtype or None
        "dtype": "dtype" # Should have been dtype or None
    },
    "inner": {}
}
signatures["torch.nn.InstanceNorm3d"] = {
    "args": {
        "num_features": "integer",
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "device": "string", # could be device, but per instruction skip
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.KLDivLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",  # Deprecated, should be float or None
        "reduce": "boolean",  # Deprecated, should be float or None
        "reduction": "string",
        "log_target": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.L1Loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.LPPool1d"] = {
    "args": {
        "norm_type": "integer",  # could be float as well, but integer seems more likely based on documentation
        "kernel_size": "integer",
        "stride": "integer" # could be tuple as well
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
signatures["torch.nn.LPPool1d_1"] = {
    "args": {
        "norm_type": "integer",  # could be float as well, but integer seems more likely based on documentation
        "kernel_size": "tuple",
        "stride": "integer" # could be tuple as well
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
signatures["torch.nn.LPPool1d_2"] = {
    "args": {
        "norm_type": "integer",  # could be float as well, but integer seems more likely based on documentation
        "kernel_size": "integer",
        "stride": "tuple" # could be tuple as well
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
signatures["torch.nn.LPPool1d_3"] = {
    "args": {
        "norm_type": "integer",  # could be float as well, but integer seems more likely based on documentation
        "kernel_size": "tuple",
        "stride": "tuple" # could be tuple as well
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
        "norm_type": "float", # could be int as well
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
        "dtype": "dtype", # Could be a dtype object
    },
    "inner": {
        "args": {
            "input": "tensor",
            "h_0": "tensor_list",
            "c_0": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LSTMCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "bias": "boolean",  # Could also be None
        "dtype": "dtype" # Added dtype, potentially missing from documentation
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hx": "tensor" # h_0
        },
        "kwargs": {
            "cx": "tensor" # c_0
        }
    }
}
signatures["torch.nn.LayerNorm"] = {
    "args": {
        "normalized_shape": "list"  # or int or torch.Size
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "bias": "boolean",
        "dtype": "dtype" # could be a torch.dtype
    },
    "inner": {},
}
signatures["torch.nn.LazyBatchNorm1d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # or None
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.LazyBatchNorm2d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float", # could be None
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.LazyInstanceNorm1d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",  # could be a tensor as well
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.LazyInstanceNorm2d"] = {
    "args": {
        # num_features can be inferred from input size, so it's not a direct argument.
    },
    "kwargs": {
        "eps": "float",  # Default: 1e-5
        "momentum": "float",  # Default: 0.1, Optional[float]
        "affine": "boolean",  # Default: False
        "track_running_stats": "boolean"  # Default: False
    },
    "inner": {},
}
signatures["torch.nn.LeakyReLU"] = {
    "args": {
        "negative_slope": "float",  # Could also be integer
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.Linear"] = {
    "args": {
        "in_features": "integer",
        "out_features": "integer"
    },
    "kwargs": {
        "bias": "boolean" # Could also be None, but documentation states default is True.
    },
    "inner": {}
}
signatures["torch.nn.LocalResponseNorm"] = {
    "args": {
        "size": "integer",
        "input": "tensor"  # documentation says input is also a parameter for the forward pass
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
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LogSoftmax"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.nn.MSELoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.MarginRankingLoss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.MaxPool1d"] = {
    "args": {
        "kernel_size": "integer",  # could also be tuple
        "stride": "integer",  # could also be tuple
        "padding": "integer",  # could also be tuple
        "dilation": "integer",  # could also be tuple
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d"] = {
    "args": {
        "kernel_size": "union[int, tuple]",  # could be int or tuple
        "stride": "union[int, tuple]",  # could be int or tuple
        "padding": "union[int, tuple]",  # could be int or tuple
        "dilation": "union[int, tuple]",  # could be int or tuple
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool3d"] = {
    "args": {
        "kernel_size": "union[integer, tuple]",  # could also be a tuple of integers
        "stride": "union[integer, tuple]",  # defaults to kernel_size, could also be a tuple of integers
        "padding": "union[integer, tuple]",  # defaults to 0, could also be a tuple of integers
        "dilation": "union[integer, tuple]",  # defaults to 1, could also be a tuple of integers
        "return_indices": "boolean",  # defaults to False
        "ceil_mode": "boolean"  # defaults to False
    },
    "kwargs": {},
    "inner": {}
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
        "kernel_size": "tuple",  # can also be integer
        "stride": "tuple",  # can also be integer, defaults to kernel_size
        "padding": "integer"  # can also be tuple
    },
    "kwargs": {
        "output_size": "tuple"  # potentially also a list or tensor, but tuple seems most common
    },
    "inner": {}
}
signatures["torch.nn.MaxUnpool3d"] = {
    "args": {
        "kernel_size": "integer",  # or tuple
        "stride": "integer",  # or tuple
        "padding": "integer",  # or tuple
    },
    "kwargs": {
        "input": "tensor",
        "indices": "tensor",
        "output_size": "tuple",  # optional
    },
    "inner": {},
}
signatures["torch.nn.Mish"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # Could also be None, but boolean is more common
    },
    "inner": {},
}
signatures["torch.nn.MultiLabelMarginLoss"] = {
    "args": {},
    "kwargs": {
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
signatures["torch.nn.MultiLabelSoftMarginLoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",  # could be None
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
        "weight": "tensor",  # Could also be None
        "size_average": "boolean",  # Deprecated
        "reduce": "boolean",  # Deprecated
        "reduction": "string",
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor", #or integer
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MultiheadAttention"] = {
    "args": {
        "embed_dim": "integer",
        "num_heads": "integer",
    },
    "kwargs": {
        "dropout": "float",
        "bias": "boolean",
        "add_bias_kv": "boolean",
        "add_zero_attn": "boolean",
        "kdim": "integer", # could be None
        "vdim": "integer", # could be None
        "batch_first": "boolean",
        "dtype": "dtype" # could be None
    },
    "inner": {
        "args": {
            "query": "tensor",
            "key": "tensor",
            "value": "tensor"
        },
        "kwargs": {
            "key_padding_mask": "tensor",  # could be None
            "need_weights": "boolean",
            "attn_mask": "tensor",  # could be None
            "average_attn_weights": "boolean",
            "is_causal": "boolean"
        }
    }
}
signatures["torch.nn.NLLLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
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
        "num_parameters": "integer", # could also be a tensor, but docstring says int
        "init": "float"
    },
    "kwargs": {},
    "inner": {}
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
        "data": "tensor"  # could also be tensor_list?
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
    "inner": {}
}
signatures["torch.nn.ParameterList.append"] = {
    "args": {
        "value": "list"  # could be any type
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.ParameterList.extend"] = {
    "args": {
        "values": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.ParameterList.extra_repr"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.PixelShuffle"] = {
    "args": {
        "upscale_factor": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"  # Could also be tensor_list
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
    "args": {
        "log_input": "boolean",
        "target": "tensor"
    },
    "kwargs": {
        "full": "boolean",
        "size_average": "boolean",  # deprecated
        "eps": "float",
        "reduce": "boolean",  # deprecated
        "reduction": "string"
    },
    "inner": {}
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
signatures["torch.nn.RNN_1"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
        "hx": "tensor"  # Could also be a list of tensors
    },
    "kwargs": {
        "num_layers": "integer",
        "nonlinearity": "string",
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hx": "tensor" #Could also be a list of tensors
        },
        "kwargs": {
            "batch_first": "boolean"
        }
    }
}
signatures["torch.nn.RNN_2"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
    },
    "kwargs": {
        "num_layers": "integer",
        "nonlinearity": "string",
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hx": "tensor" #Could also be a list of tensors
        },
        "kwargs": {
            "batch_first": "boolean"
        }
    }
}
signatures["torch.nn.RNN_3"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
        "num_layers": "integer"
    },
    "kwargs": {
        "nonlinearity": "string",
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "hx": "tensor"  # Could also be a list of tensors
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {
            "batch_first": "boolean"
        }
    }
}
signatures["torch.nn.RNNCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "bias": "boolean",  # Could also be None
        "nonlinearity": "string"  # Can be 'tanh' or 'relu'
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hidden": "tensor" # Or None
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RReLU"] = {
    "args": {
        "lower": "float",  # Could also be integer
        "upper": "float",  # Could also be integer
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.ReLU"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean" # could also be None
    },
    "inner": {}
}
signatures["torch.nn.ReLU6"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.ReflectionPad1d"] = {
    "args": {
        "padding": "integer" # or tuple
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
            "input": "tensor" # Could also be tensor_list
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
            "input": "tensor" # Could also be tensor_list
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
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad3d_2"] = {
    "args": {
        "padding": "tuple" # Should be a tuple of 6 integers, but we'll just mark it as tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
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
            "input": "tensor" # Could also be tensor_list
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
            "input": "tensor" # Could also be tensor_list
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad2d_1"] = {
    "args": {
        "padding": "integer"  # could also be tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
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
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad3d_1"] = {
    "args": {
        "padding": "integer"  # could also be tuple
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
        "inplace": "boolean"  # Should it be boolean or integer? boolean seems most reasonable
    },
    "inner": {}
}
signatures["torch.nn.SiLU"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # Could also be None, but boolean is more common
    },
    "inner": {},
}
signatures["torch.nn.Sigmoid"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list, but documentation says element-wise so single tensor is more likely
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.SmoothL1Loss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string",
        "beta": "float"
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
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.nn.Softmax"] = {
    "args": {
        "input": "tensor" # could also be sparse tensor
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
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
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.nn.Softplus"] = {
    "args": {},
    "kwargs": {
        "beta": "float",
        "threshold": "float"
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
        "input": "tensor" # could also be tensor_list, but documentation says ()*(*)
    },
    "kwargs": {
        "lambd": "float" # lambda is a float
    },
    "inner": {}
}
signatures["torch.nn.Softsign"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.SyncBatchNorm"] = {
    "args": {
        "num_features": "integer",
        "eps": "float",
        "momentum": "float", # Could be None
        "affine": "boolean",
        "track_running_stats": "boolean",
        "process_group": "list", # Could be Any
        "device": "string",
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.SyncBatchNorm.convert_sync_batchnorm"] = {
    "args": {
        "module": "tensor" # Actually nn.Module
    },
    "kwargs": {
        "process_group": "list" # Could be Any
    },
    "inner": {}
}
signatures["torch.nn.Tanh"] = {
    "args": {
        "input": "tensor"  # Input can be any shape tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.Tanhshrink"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list, but documentation says any number of dimensions, suggesting a single tensor
    },
    "kwargs": {},
    "inner": {}
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
            "input": "tensor"  # Could also be tensor_list, but documentation implies a single tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Transformer"] = {
    "args": {
        "d_model": "integer",
        "nhead": "integer",
        "num_encoder_layers": "integer",
        "num_decoder_layers": "integer",
        "dim_feedforward": "integer",
        "dropout": "float",
        "src": "tensor", # Should be tensor or tensor_list
        "tgt": "tensor" # Should be tensor or tensor_list
    },
    "kwargs": {
        "activation": "string",  # or callable
        "custom_encoder": "list", # Or Any, but list seems most appropriate
        "custom_decoder": "list", # Or Any, but list seems most appropriate
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
        "memory_is_causal": "boolean",
    },
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.nn.Transformer_generate_square_subsequent_mask"] = {
    "args": {
        "sz": "integer",
    },
    "kwargs": {
        "device": "tensor", # Should be device instead of tensor
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
            "src_mask": "tensor", #optional
            "src_key_padding_mask": "tensor", #optional
            "is_causal": "boolean" #optional
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
        "margin": "float",
        "p": "integer",
        "eps": "float",
        "swap": "boolean",
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.Unflatten_1"] = {
    "args": {
        "dim": "integer" # Could also be a tuple of integers
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Unflatten_2"] = {
    "args": {
        "dim": "tuple" # Could also be a tuple of integers
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
        "kernel_size": "tuple",  # Could be int or tuple
        "input": "tensor"
    },
    "kwargs": {
        "dilation": "tuple",  # Could be int or tuple
        "padding": "tuple",  # Could be int or tuple
        "stride": "tuple"  # Could be int or tuple
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
            "input": "tensor"  # should be tensor
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
            "input": "tensor"  # should be tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ZeroPad2d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.ZeroPad2d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {}
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
        "output_size": "tuple"  # or integer, creating separate signatures
    },
    "kwargs": {
        "stride": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.adaptive_avg_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.adaptive_avg_pool3d"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # or list, but tuple seems more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool1d"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "integer", # Could be a tuple, adding signature for that in the next entry
        "dilation": "integer", # Could be a tuple, adding signature for that in the next entry
        "padding": "integer" # Could be a tuple, adding signature for that in the next entry
    },
    "inner": {},
}

signatures["torch.nn.functional.adaptive_max_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "dilation": "integer",
        "padding": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.adaptive_max_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "dilation": "tuple",
        "padding": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.adaptive_max_pool1d_3"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "dilation": "integer",
        "padding": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.adaptive_max_pool1d_4"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "dilation": "tuple",
        "padding": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.adaptive_max_pool1d_5"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "dilation": "integer",
        "padding": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.adaptive_max_pool1d_6"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "dilation": "tuple",
        "padding": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.adaptive_max_pool1d_7"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "dilation": "tuple",
        "padding": "tuple"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool2d"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # could also be integer
    },
    "kwargs": {
        "return_indices": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool3d"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # could also be integer
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.affine_grid"] = {
    "args": {
        "theta": "tensor",
        "size": "tuple"  # could also be list
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
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
        "count_include_pad": "boolean"
    },
    "kwargs": {
        "divisor": "integer",  # Should it be float?
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.functional.avg_pool2d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",  # or tuple
        "stride": "integer",  # or tuple
        "padding": "integer",  # or tuple or string
        "dilation": "integer",  # or tuple
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "divisible": "boolean",  # should it be integer?
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"  # or tuple
    },
    "kwargs": {
        "stride": "integer",  # or tuple
        "padding": "integer",  # or tuple
        "dilation": "integer",  # or tuple
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {}
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
        "training": "boolean",  # Can also be a boolean
        "momentum": "float",
        "eps": "float",
        "affine": "boolean",
        "moving_average": "float", # This seems redundant with momentum
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
    "inner": {}
}
signatures["torch.nn.functional.binary_cross_entropy_with_logits_1"] = {
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
signatures["torch.nn.functional.binary_cross_entropy_with_logits_2"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor", # Could be None
        "reduction": "string"
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
        "bias": "tensor",  # can be None
        "stride": "integer",  # can be a tuple
        "padding": "string",  # can be an integer or a tuple
        "dilation": "integer",  # can be a tuple
        "groups": "integer"
    },
    "inner": {}
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
        "stride": "integer",
        "padding": "string", # can also be integer or tuple
        "dilation": "integer", # can also be tuple
        "groups": "integer"
    },
    "kwargs": {
        "bias": "tensor",
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
        "output_size": "integer", # Could be a tuple, adding signature for that later
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor",
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "output_size": "tuple",
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor",  # Could be None
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",  # Could be None
        "output_size": "tuple" # Might be deprecated
    },
    "inner": {}
}
signatures["torch.nn.functional.conv_transpose3d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor",
        "stride": "tuple", # Could also be integer
        "padding": "tuple", # Could also be integer
        "dilation": "tuple", # Could also be integer
        "groups": "integer",
    },
    "kwargs": {
        "output_size": "tuple", # Should this be integer?
        "output_padding": "tuple", # Could also be integer
    },
    "inner": {},
}
signatures["torch.nn.functional.cosine_embedding_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float", # Could also be integer, but float seems more general
        "reduction": "string" # Should be string for 'mean', 'sum', or 'none'
    },
    "inner": {},
}
signatures["torch.nn.functional.cosine_similarity"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor"
    },
    "kwargs": {
        "dim": "integer",  # Could also be -1, but integer is most accurate
        "eps": "float"
    },
    "inner": {}
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
        "label_smoothing": "float" # might be None
    },
    "inner": {},
}
signatures["torch.nn.functional.cross_entropy_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
    },
    "kwargs": {
        "weight": "tensor",  # could be None
        "ignore_index": "integer",
        "reduction": "string",
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
        "blank": "integer", # Should it be float?
        "reduction": "string" # or "enum"
    },
    "inner": {},
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
        "inplace": "boolean" # Could also be None, but boolean is more likely based on description
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
        "embedding_bag": "tensor",
        "indices": "tensor",
        "offsets": "tensor",
        "max_norm": "float", # Could also be None
        "norm_type": "integer", # Could be 2, 3, etc.
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "kwargs": {
        "dtype": "dtype" # Should it be torch.dtype?
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
        "training": "boolean" # could also be a boolean
    },
    "inner": {}
}
signatures["torch.nn.functional.fold"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple", # or list, but tuple seems more common for sizes
        "kernel_size": "tuple", # or list
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
        "output_size": "tuple"  # Could also be integer, but tuple is more general
    },
    "kwargs": {
        "dilation": "list", # Could also be integer
        "padding": "list", # Could also be integer
        "stride": "list", # Could also be integer
        "return_indices": "boolean",
        "correct_border_mode": "boolean"
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
        "dim": "integer"
    },
    "kwargs": {
        "momentum": "float",
        "affine": "boolean",
        "out": "tensor" # Could be tensor or None
    },
    "inner": {}
}
signatures["torch.nn.functional.gumbel_softmax"] = {
    "args": {
        "input": "tensor",
        "tau": "float",
        "hard": "boolean"
    },
    "kwargs": {
        "dim": "integer" # Could be -1 as well. Choosing integer.
    },
    "inner": {},
}
signatures["torch.nn.functional.hardshrink"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "lambd": "float"  # Could also be integer, but float is more common for shrinkage parameters
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
    "inner": {},
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
        "dist_function": "string" # Should it be callable?
    },
    "inner": {},
}
signatures["torch.nn.functional.huber_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "delta": "float", # Could also be a tensor
        "reduction": "string" # Should be enum. Can be 'none', 'mean', 'sum'
    },
    "inner": {}
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
        "eps": "float",  # Could be a float or a DType
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
        "log_target": "boolean", # Could be tensor too, but docstring says boolean
        "reduction": "string", # Should be string, options are 'none', 'sum', 'mean', 'batchmean'
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
        "reduction": "string", # could be 'none', 'mean', 'sum'
        "size_average": "boolean" # Deprecated
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
        "input": "tensor",
    },
    "kwargs": {
        "negative_slope": "float",  # Could also be an integer
    },
    "inner": {},
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
        "k": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.log_softmax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.logsigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
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
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.lp_pool2d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer", # Could also be tuple
        "stride": "integer", # Could also be tuple
        "padding": "integer", # Could also be tuple
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
        "label": "tensor"  # could also be integer, but tensor is more general
    },
    "kwargs": {
        "margin": "float",
        "reduction": "string"  # Could also be 'none', but string is more general
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
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"  # Could also be tensor_list, but tensor seems more common.
    },
    "kwargs": {},
    "inner": {}
}
