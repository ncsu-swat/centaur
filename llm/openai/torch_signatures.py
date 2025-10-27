signatures = {}
signatures["torch.DoubleStorage"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ShortStorage"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.abs"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.abs_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.absolute"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.acos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.acos"] = {
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
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.add"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addbmm_1"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float"
    },
    "inner": {},
}
signatures["torch.addcdiv_1"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addcmul_1"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",
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
        "beta": "float",
        "alpha": "float",
        "out_dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.addmv_1"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "beta": "integer",
        "alpha": "integer"
    },
    "inner": {}
}
signatures["torch.addmv_2"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float"
    },
    "inner": {}
}
signatures["torch.addmv__1"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "beta": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addr_1"] = {
    "args": {
        "input": "tensor",
        "vec1": "tensor",
        "vec2": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float"
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
        "dim": "integer"
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
    "inner": {},
}
signatures["torch.amax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.amin_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.amin_2"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.amin_3"] = {
    "args": {
        "input": "tensor",
        "dim": "None"
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.aminmax_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.angle"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.any_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.any_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.arange_1"] = {
    "args": {
        "start": "float",
        "end": "float",
        "step": "float"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {}
}

signatures["torch.arange_2"] = {
    "args": {
        "start": "integer",
        "end": "integer",
        "step": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {}
}

signatures["torch.arange_3"] = {
    "args": {
        "start": "integer",
        "end": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {}
}

signatures["torch.arange_4"] = {
    "args": {
        "start": "float",
        "end": "float"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
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
signatures["torch.arccos"] = {
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
    "kwargs": {},
    "inner": {}
}
signatures["torch.arcsin_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.arcsin_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arcsinh_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arcsinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arctan_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.arctan_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arctanh"] = {
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
    "kwargs": {},
    "inner": {},
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
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.argmin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.argsort"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "descending": "boolean",
        "stable": "boolean"
    },
    "inner": {},
}
signatures["torch.argwhere_1"] = {
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
        "offset": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.as_strided_copy"] = {
    "args": {
        "input": "tensor",
        "size": "tuple",
        "stride": "tuple",
        "storage_offset": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.as_tensor"] = {
    "args": {
        "input": "tensor",
        "dtype": "dtype"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.asarray"] = {
    "args": {
        "obj": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "torch.device",
        "copy": "boolean",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.asin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.asin__1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.asinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.asinh__1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atan_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.atan2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.atan_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.atanh_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.atanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_1d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.atleast_2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_3d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.autocast"] = {
    "args": {
        "device": "string",
        "enabled": "boolean"
    },
    "kwargs": {
        "dtype": "dtype",
        "cache_enabled": "boolean"
    },
    "inner": {},
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
signatures["torch.baddbmm_1"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float",
        "out_dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.bartlett_window"] = {
    "args": {
        "window_length": "integer"
    },
    "kwargs": {
        "period": "float",
        "endpoint": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.bilinear"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bincount_1"] = {
    "args": {
        "input": "tensor",
        "weights": "tensor"
    },
    "kwargs": {
        "minlength": "integer"
    },
    "inner": {},
}

signatures["torch.bincount_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
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
    "kwargs": {},
    "inner": {},
}
signatures["torch.bitwise_or"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bitwise_right_shift"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bitwise_xor"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.blackman_window"] = {
    "args": {
        "window_length": "integer"
    },
    "kwargs": {
        "beta": "float",
        "dtype": "dtype",
        "layout": "tensor_list",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.block_diag"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bmm_1"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype"
    },
    "inner": {}
}

signatures["torch.bmm_2"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.bmm_3"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.broadcast_shapes"] = {
    "args": {
        "shape1": "tuple",
        "shape2": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.broadcast_tensors"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "group": "string",
        "async_op": "boolean"
    },
    "inner": {},
}
signatures["torch.broadcast_to"] = {
    "args": {
        "input": "tensor",
        "size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bucketize"] = {
    "args": {
        "input": "tensor",
        "boundary": "tensor"
    },
    "kwargs": {
        "out_int32": "boolean",
        "right": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.cartesian_prod"] = {
    "args": {
        " tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cat_1"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "dim": "integer",
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
        "p": "float",
        "compute_mode": "string"
    },
    "inner": {}
}
signatures["torch.ceil"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ceil_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.celu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        " inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.chain_matmul"] = {
    "args": {
        "input": "tensor",
        "matrices": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.channel_shuffle"] = {
    "args": {
        "input": "tensor",
        "groups": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cholesky"] = {
    "args": {
        "input": "tensor",
        "upper": "boolean"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.cholesky_inverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "upper": "boolean"
    },
    "inner": {}
}
signatures["torch.cholesky_solve"] = {
    "args": {
        "input": "tensor",
        "L": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.chunk_1"] = {
    "args": {
        "input": "tensor",
        "chunks": "integer"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {}
}
signatures["torch.clamp_1"] = {
    "args": {
        "input": "tensor",
        "min": "float",
        "max": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.clamp_2"] = {
    "args": {
        "input": "tensor",
        "min": "tensor",
        "max": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.clamp_max"] = {
    "args": {
        "input": "tensor",
        "max": "float"
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
        "input": "tensor",
        "min": "tensor",
        "max": "tensor"
    },
    "kwargs": {},
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
        " tensors": "tensor_list"
    },
    "kwargs": {
        "dim": "integer"
    },
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
        "real": "tensor",
        "imag": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.concat"] = {
    "args": {
        "tensors": "tensor_list",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.concatenate_1"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "axis": "integer",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.conj_1"] = {
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
    "inner": {},
}
signatures["torch.conj_physical__1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.copysign"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.corrcoef"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": " tensor"
    },
    "inner": {}
}
signatures["torch.cos"] = {
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
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.cosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.count_nonzero_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.count_nonzero_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cross_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.crow_indices_copy"] = {
    "args": {
        "input": "tensor",
        "crow_indices": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cudnn_affine_grid_generator"] = {
    "args": {
        "batch": "integer",
        "channel": "integer",
        "height": "integer",
        "width": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cummax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.cummin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {},
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
    "inner": {},
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
    "inner": {},
}
signatures["torch.deg2rad"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.device"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.diag_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "diagonal": "integer"
    },
    "inner": {}
}
signatures["torch.diag_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "diagonal": "integer",
 "out": "tensor"
    },
    "inner": {}
}
signatures["torch.diag_embed"] = {
    "args": {
        "input": "tensor",
        "k": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.diagflat"] = {
    "args": {
        "input": "tensor",
        "offset": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.diagonal"] = {
    "args": {
        "input": "tensor",
        "offset": "integer"
    },
    "kwargs": {
        "dim1": "integer",
        "dim2": "integer"
    },
    "inner": {},
}
signatures["torch.diff"] = {
    "args": {
        "input": "tensor",
        "n": "integer",
        "dim": "integer"
    },
    "kwargs": {
        "prepend": "tensor",
        "append": "tensor",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.digamma"] = {
    "args": {
        "input": "tensor"
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
        "p": "float"
    },
    "inner": {}
}
signatures["torch.div"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rounding_mode": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.divide_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rounding_mode": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.dot"] = {
    "args": {
        "input": "tensor",
        "tensor": "tensor"
    },
    "kwargs": {
        "out": " tensor"
    },
    "inner": {},
}
signatures["torch.dsplit"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.dstack"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.einsum"] = {
    "args": {
        "equation": "string",
        "operands": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.empty_1"] = {
    "args": {
        "size": "tuple"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "torch.memory_format"
    },
    "inner": {}
}
signatures["torch.empty_2"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "torch.memory_format"
    },
    "inner": {}
}
signatures["torch.empty_like_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "pin_memory": "boolean"
    },
    "inner": {},
}

signatures["torch.empty_like_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.empty_strided"] = {
    "args": {
        "size": "tuple",
        "stride": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.eq"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
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
    "inner": {},
}
signatures["torch.erf_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.erf__1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.erfc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.erfc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.exp"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
        "dims": "integer"
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
    "inner": {}
}
signatures["torch.eye"] = {
    "args": {
        "n": "integer",
        "m": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.fake_quantize_per_channel_affine"] = {
    "args": {
        "input": "tensor",
        "scale": "tensor",
        "zero_point": "tensor",
        "axis": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fake_quantize_per_tensor_affine"] = {
    "args": {
        "input": "tensor",
        "scale": "float",
        "zero_point": "integer",
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.fft.fft_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string"
    },
    "inner": {}
}
signatures["torch.fft.fft2_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.fftfreq_1"] = {
    "args": {
        "n": "integer",
        "d": "float"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.fft.fftfreq_2"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "d": "float",
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.fft.fftn_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.fftshift_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fft.fftshift_2"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fft.hfft_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string"
    },
    "inner": {}
}
signatures["torch.fft.hfft_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.ifft_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string"
    },
    "inner": {}
}
signatures["torch.fft.ifft2_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.ifftn"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.ifftshift"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fft.ihfft_1"] = {
    "args": {
        "input": "tensor",
        "n": "integer",
        "dim": "integer",
        "norm": "string"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.irfft_1"] = {
    "args": {
        "input": "tensor",
        "n": "integer"
    },
    "kwargs": {
        "dim": "integer",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.irfft_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.irfft2_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string"
    },
    "inner": {}
}
signatures["torch.fft.irfft2_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
 "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.irfftn_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.irfftn_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.rfft"] = {
    "args": {
        "input": "tensor",
        "n": "integer",
        "dim": "integer",
        "norm": "string"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.rfft2_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string"
    },
    "inner": {}
}
signatures["torch.fft.rfftfreq"] = {
    "args": {
        "n": "integer",
        "d": "float"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.fft.rfftn"] = {
    "args": {
        "input": "tensor",
        "s": "tuple",
        "dim": "tuple",
        "norm": "string"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.finfo"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fix"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.flatten"] = {
    "args": {
        "input": "tensor",
        "start_dim": "integer",
        "end_dim": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.flip_1"] = {
    "args": {
        "input": "tensor",
        "dims": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.flip_2"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.fliplr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
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
        " exponent": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": " tensor"
    },
    "inner": {},
}
signatures["torch.floor_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor_divide"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
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
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fmin"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.frac"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.frexp_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.from_dlpack"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.from_numpy_1"] = {
    "args": {
        "array": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.full"] = {
    "args": {
        "size": "tuple",
        "fill_value": "float"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.full_like"] = {
    "args": {
        "input": "tensor",
        "fill_value": "float"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "integer",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.gather"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor"
    },
    "kwargs": {
        "sparse_grad": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.gcd__1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.gcd__2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ge"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
    "inner": {}
}
signatures["torch.get_deterministic_debug_mode"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_device"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_file_path"] = {
    "args": {
        "file": "string"
    },
    "kwargs": {},
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
        "spacing": "float",
        "dim": "integer",
        "edge_order": "integer"
    },
    "inner": {}
}
signatures["torch.greater_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.greater_equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.gt"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.hamming_window"] = {
    "args": {
        "window_length": "integer"
    },
    "kwargs": {
        "period": "float",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.hann_window"] = {
    "args": {
        "window_length": "integer"
    },
    "kwargs": {
        "period": "float",
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
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
    "inner": {},
}
signatures["torch.histc"] = {
    "args": {
        "input": "tensor",
        "bin": "integer",
        "min": "float",
        "max": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.histogram"] = {
    "args": {
        "input": "tensor",
        "bins": "integer"
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
        "indices_or_sections": "integer"
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
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.hstack"] = {
    "args": {
        "tensors": "tensor_list"
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
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.i0"] = {
    "args": {
        "input": "tensor"
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
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.iinfo"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.imag"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.index_add"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.index_copy"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.index_put_1"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor_list"
    },
    "kwargs": {
        "value": "tensor"
    },
    "inner": {},
}

signatures["torch.index_put_2"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor_list"
    },
    "kwargs": {
        "value": "float"
    },
    "inner": {},
}
signatures["torch.index_select"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.inner"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.inverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
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
    "inner": {},
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
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_grad_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_inference"] = {
    "args": {
        "input": "tensor"
    },
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
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_same_size"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
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
        "input": "tensor"
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
signatures["torch.isin_1"] = {
    "args": {
        "elements": "tensor",
        "test_elements": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean"
    },
    "inner": {}
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
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.isposinf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.isreal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.CompilationUnit"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "other": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.jit.Error"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float"
    },
    "inner": {},
}
signatures["torch.jit.ScriptWarning"] = {
    "args": {
        "message": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.enable_onednn_fusion"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.ignore"] = {
    "args": {
        "drop": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.is_scripting"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.is_tracing"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.isinstance"] = {
    "args": {
        "obj": "tensor",
        "target_type": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.optimized_execution"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "optimization": "boolean",
        "backend": "string"
    },
    "inner": {},
}
signatures["torch.jit.script_if_tracing"] = {
    "args": {
        "condition": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.set_fusion_strategy"] = {
    "args": {
        "strategy": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.set_module"] = {
    "args": {
        "module": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.strict_fusion"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.jit.wait"] = {
    "args": {
        "future": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.kaiser_window"] = {
    "args": {
        "window": "tensor",
        "beta": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.kron"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.kthvalue"] = {
    "args": {
        "input": "tensor",
        "k": "integer"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean"
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
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.le"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lerp"] = {
    "args": {
        "input": "tensor",
        "end": "tensor",
        "weight": "float"
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
    "kwargs": {},
    "inner": {}
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
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.cholesky_1"] = {
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
        "input": "tensor"
    },
    "kwargs": {
        "upper": "boolean"
    },
    "inner": {}
}
signatures["torch.linalg.det_1"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor"
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
    "inner": {},
}
signatures["torch.linalg.eigh"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "UPLO": "string",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.linalg.eigvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.eigvalsh"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "UPLO": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.householder_product"] = {
    "args": {
        "tau": "tensor",
        "v": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.inv_1"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.lstsq_1"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "rcond": "float",
        "driver": "string"
    },
    "inner": {}
}
signatures["torch.linalg.lu_1"] = {
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
        "input": "tensor",
        "ord": "string",
        "dim": "tuple",
        "keepdim": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.linalg.matrix_power"] = {
    "args": {
        "input": "tensor",
        "n": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.matrix_rank_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "tol": "float",
        "hermitian": "boolean"
    },
    "inner": {},
}

signatures["torch.linalg.matrix_rank_2"] = {
    "args": {
        "input": "tensor",
        "tol": "float"
    },
    "kwargs": {
        "hermitian": "boolean"
    },
    "inner": {},
}
signatures["torch.linalg.multi_dot"] = {
    "args": {
        "matrices": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.norm_1"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "ord": "integer",
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.linalg.norm_2"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "ord": "float",
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.linalg.norm_3"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "ord": "string",
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.linalg.norm_4"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "ord": "integer",
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.linalg.norm_5"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "ord": "float",
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.linalg.norm_6"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "ord": "string",
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.linalg.pinv_1"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "atol": "float",
        "rtol": "float",
        "hermitian": "boolean"
    },
    "inner": {},
}
signatures["torch.linalg.qr"] = {
    "args": {
        "A": "tensor",
        "mode": "string"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.linalg.slogdet"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tuple"
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
    "inner": {},
}
signatures["torch.linalg.solve_ex"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.solve_triangular"] = {
    "args": {
        "upper": "boolean",
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.svd"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "full_matrices": "boolean",
        "driver": "string",
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.linalg.svdvals_1"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "driver": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.linalg.tensorinv"] = {
    "args": {
        "A": "tensor",
        "ind": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.tensorsolve"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "dims": "tuple",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.vecdot"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.vector_norm"] = {
    "args": {
        "input": "tensor",
        "ord": "float",
        "dim": "integer",
        "keepdim": "boolean"
    },
    "kwargs": {
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.linspace_1"] = {
    "args": {
        "start": "float",
        "end": "float",
        "steps": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.log"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.log10"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.log_"] = {
    "args": {
        "input": "tensor"
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
        "out": "tensor"
    },
    "inner": {},
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
    "inner": {},
}
signatures["torch.logical_not"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logical_or"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logical_xor"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor",
        "eps": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logit__1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.logit__2"] = {
    "args": {
        "input": "tensor",
        " eps": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logspace"] = {
    "args": {
        "start": "float",
        "end": "float",
        "steps": "integer"
    },
    "kwargs": {
        "base": "float",
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.logsumexp"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.lt"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.lu_solve"] = {
    "args": {
        "b": "tensor",
        " LU": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.lu_unpack"] = {
    "args": {
        "LU": "tensor",
        "pivot": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.manual_seed"] = {
    "args": {
        "seed": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.masked_scatter"] = {
    "args": {
        "input": "tensor",
        "mask": " tensor",
        "source": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.masked_select"] = {
    "args": {
        "input": "tensor",
        "mask": " tensor"
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
        "out": "tensor"
    },
    "inner": {},
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
        "n": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.max_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.max_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "keepdim": "boolean"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {}
}

signatures["torch.max_3"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.maximum"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mean_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}

signatures["torch.mean_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.median_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.median_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.meshgrid"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "indexing": "string"
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
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"
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
signatures["torch.minimum_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.miopen_batch_norm"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor",
        "running_mean": "tensor",
        "running_var": "tensor"
    },
    "kwargs": {
        "training": "boolean",
        "momentum": "float",
        "eps": "float"
    },
    "inner": {},
}
signatures["torch.mm"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.moveaxis_1"] = {
    "args": {
        "input": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.moveaxis_2"] = {
    "args": {
        "input": "tensor",
        "source": "tuple",
        "destination": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.movedim_1"] = {
    "args": {
        "input": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.movedim_2"] = {
    "args": {
        "input": "tensor",
        "source": "tuple",
        "destination": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.msort"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mul_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mul_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
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
        "out": "tensor"
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
        "input": "tensor"
    },
    "kwargs": {
        "nan": "float",
        "posinf": "float",
        "neginf": "float"
    },
    "inner": {},
}
signatures["torch.nanmean"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "keepdim": "boolean"
    },
    "kwargs": {
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
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"
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
        "dim": "integer"
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
        "start": "integer",
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
    "inner": {},
}
signatures["torch.native_channel_shuffle"] = {
    "args": {
        "input": "tensor",
        "groups": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.native_dropout"] = {
    "args": {
        "input": "tensor",
        "p": "float",
        "training": "boolean"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.ne"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.neg"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.negative"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.negative__1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nextafter"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.AdaptiveAvgPool1d_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.AdaptiveAvgPool1d_2"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveAvgPool2d_1"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.AdaptiveAvgPool2d_2"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.AdaptiveAvgPool2d_3"] = {
    "args": {
        "output_size": "None"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveAvgPool3d_1"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.AdaptiveAvgPool3d_2"] = {
    "args": {
        "output_size": "integer"
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
        "cutoffs": "list"
    },
    "kwargs": {
        "div_value": "float",
        "head_bias": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveMaxPool1d_1"] = {
    "args": {
        "output_size": "integer"
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
signatures["torch.nn.AdaptiveMaxPool1d_2"] = {
    "args": {
        "output_size": "tuple"
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
signatures["torch.nn.AdaptiveMaxPool2d_1"] = {
    "args": {
        "output_size": "tuple"
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

signatures["torch.nn.AdaptiveMaxPool2d_2"] = {
    "args": {
        "output_size": "integer"
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

signatures["torch.nn.AdaptiveMaxPool2d_3"] = {
    "args": {
        "output_size": "tuple"
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
signatures["torch.nn.AdaptiveMaxPool3d_1"] = {
    "args": {
        "output_size": "tuple"
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
signatures["torch.nn.AdaptiveMaxPool3d_2"] = {
    "args": {
        "output_size": "integer"
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
signatures["torch.nn.AdaptiveMaxPool3d_3"] = {
    "args": {
        "output_size": "tuple"
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
        "p": "float",
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool1d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
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
signatures["torch.nn.AvgPool1d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
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
        "kernel_size": "integer",
        "stride": "integer"
    },
    "kwargs": {
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool3d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool3d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.BCELoss"] = {
    "args": {
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.BCEWithLogitsLoss_1"] = {
    "args": {
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string",
        "pos_weight": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.BatchNorm1d_1"] = {
    "args": {
        "num_features": "integer",
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "kwargs": {
        "device": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.BatchNorm2d_1"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.BatchNorm3d_1"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.BatchNorm3d_2"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "device": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Bilinear"] = {
    "args": {
        "in1_features": "integer",
        "in2_features": "integer",
        "out_features": "integer"
    },
    "kwargs": {
        "bias": "boolean"
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
    "args": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.CTCLoss"] = {
    "args": {
        "log_probs": "tensor",
        "targets": "tensor",
        "input_lengths": "tensor",
        "target_lengths": "tensor"
    },
    "kwargs": {
        "blank": "integer",
        "reduction": "string",
        "zero_infinity": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.CircularPad1d_1"] = {
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
        "value": "float"
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
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad2d_1"] = {
    "args": {
        "padding": "integer",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad2d_2"] = {
    "args": {
        "padding": "tuple",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad3d_1"] = {
    "args": {
        "padding": "integer",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad3d_2"] = {
    "args": {
        "padding": "tuple",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Conv1d_1"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Conv1d_2"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Conv2d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Conv3d_1"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string"
    },
    "inner": {}
}
signatures["torch.nn.Conv3d_2"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string"
    },
    "inner": {}
}
signatures["torch.nn.ConvTranspose1d_1"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "integer",
        "padding_mode": "string"
    },
    "inner": {}
}
signatures["torch.nn.ConvTranspose2d_1"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "integer"
    },
    "inner": {}
}
signatures["torch.nn.ConvTranspose2d_2"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "tuple"
    },
    "inner": {}
}
signatures["torch.nn.ConvTranspose2d_3"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {
            "output_size": "list"
        }
    }
}
signatures["torch.nn.ConvTranspose2d_4"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "tuple"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {
            "output_size": "list"
        }
    }
}
signatures["torch.nn.ConvTranspose3d_1"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "integer"
    },
    "inner": {}
}
signatures["torch.nn.ConvTranspose3d_2"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "tuple"
    },
    "inner": {}
}
signatures["torch.nn.CosineEmbeddingLoss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.CosineSimilarity"] = {
    "args": {
        "dim": "integer",
        "eps": "float"
    },
    "kwargs": {},
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
        "weight": "tensor",
        "size_average": "boolean",
        "ignore_index": "integer",
        "reduce": "boolean",
        "reduction": "string",
        "label_smoothing": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Dropout"] = {
    "args": {
        "p": "float",
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Dropout2d_1"] = {
    "args": {
        "p": "float",
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Dropout3d_1"] = {
    "args": {
        "p": "float"
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
signatures["torch.nn.ELU"] = {
    "args": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
```python
signatures["torch.nn.Embedding_1"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
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
signatures["torch.nn.Embedding_2"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer",
        "padding_idx": "integer"
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.Embedding_3"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer",
        "padding_idx": "integer"
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean",
        "weight": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.Embedding_4"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean",
        "weight": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.Embedding_5"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_6"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_7"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_8"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_9"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_10"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_11"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_12"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_13"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_14"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_15"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_16"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_17"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_18"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_19"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_20"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_21"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_22"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_23"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_24"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_25"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_26"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_27"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_28"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_29"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_30"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_31"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_32"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_33"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_34"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_35"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_36"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_37"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_38"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_39"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_40"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_41"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_42"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_43"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_44"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_45"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_46"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_47"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_48"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_49"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_50"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_51"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_52"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_53"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_54"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_55"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_56"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_57"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_58"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_59"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_60"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_61"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_62"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_63"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_64"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_65"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_66"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_67"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_68"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_69"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_70"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_71"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_72"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_73"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_74"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_75"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_76"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_77"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_78"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_79"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_80"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_81"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_82"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_83"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_84"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_85"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_86"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_87"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_88"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_89"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_90"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_91"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_92"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_93"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_94"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_95"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_96"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_97"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_98"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_99"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_100"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_101"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_102"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_103"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_104"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_105"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_106"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_107"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_108"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_109"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_110"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_111"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_112"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_113"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_114"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_115"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_116"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_117"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_118"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_119"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_120"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_121"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_122"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_123"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_124"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_125"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_126"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_127"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_128"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_129"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_130"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_131"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_132"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_133"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_134"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_135"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_136"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_137"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_138"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_139"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_140"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_141"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_142"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_143"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_144"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_145"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_146"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_147"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_148"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_149"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_150"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_151"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_152"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_153"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_154"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_155"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_156"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_157"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_158"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_159"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_160"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_161"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_162"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_163"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_164"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_165"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_166"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_167"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_168"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_169"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_170"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_171"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_172"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_173"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_174"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_175"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_176"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_177"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_178"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_179"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_180"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_181"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_182"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_183"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_184"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_185"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_186"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_187"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_188"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_189"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_190"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_191"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_192"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_193"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_194"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_195"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_196"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_197"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_198"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_199"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_200"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_201"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_202"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_203"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_204"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_205"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_206"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_207"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_208"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_209"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_210"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_211"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_212"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_213"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_214"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_215"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_216"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_217"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_218"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_219"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_220"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_221"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_222"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_223"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_224"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_225"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_226"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_227"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_228"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_229"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_230"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_231"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_232"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_233"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_234"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_235"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_236"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_237"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_238"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_239"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_240"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_241"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_242"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_243"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_244"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_245"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_246"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_247"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_248"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_249"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor_list"        },
        "kwargs": {}
    }
}
signatures["torch.nn.Embedding_250"]
signatures["torch.nn.EmbeddingBag_1"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "include_last_offset": "boolean",
        "padding_idx": "integer"
    },
    "inner": {}
}
signatures["torch.nn.EmbeddingBag_2"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer",
        "weight": "tensor"
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "include_last_offset": "boolean",
        "padding_idx": "integer"
    },
    "inner": {}
}
signatures["torch.nn.EmbeddingBag_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "offsets": "tensor",
        "per_sample_weights": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.EmbeddingBag_4"] = {
    "args": {
        "embeddings": "tensor"
    },
    "kwargs": {
        "freeze": "boolean",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "include_last_offset": "boolean",
        "padding_idx": "integer"
    },
    "inner": {}
}
signatures["torch.nn.FeatureAlphaDropout"] = {
    "args": {
        "p": "float",
        "inplace": "boolean"
    },
    "kwargs": {},
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
        "start_dim": "integer",
        "end_dim": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.Fold"] = {
    "args": {
        "output_size": "tuple",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "tuple"
    },
    "inner": {}
}
signatures["torch.nn.FractionalMaxPool2d_1"] = {
    "args": {
        "kernel_size": "integer",
        "output_size": "integer"
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

signatures["torch.nn.FractionalMaxPool2d_2"] = {
    "args": {
        "kernel_size": "integer",
        "output_ratio": "float"
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

signatures["torch.nn.FractionalMaxPool2d_3"] = {
    "args": {
        "kernel_size": "tuple",
        "output_size": "tuple"
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

signatures["torch.nn.FractionalMaxPool2d_4"] = {
    "args": {
        "kernel_size": "tuple",
        "output_ratio": "tuple"
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
signatures["torch.nn.FractionalMaxPool3d_1"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "output_size": "tuple",
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.FractionalMaxPool3d_2"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "output_size": "integer",
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.FractionalMaxPool3d_3"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.FractionalMaxPool3d_4"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "output_ratio": "float",
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.GELU"] = {
    "args": {
        "approximate": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.GLU"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {}
}
signatures["torch.nn.GRU_1"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
        "num_layers": "integer"
    },
    "kwargs": {
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.GRU_2"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
        "num_layers": "integer"
    },
    "kwargs": {
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "device": "string",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.GRUCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "bias": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hidden": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.GaussianNLLLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
        "var": "tensor"
    },
    "kwargs": {
        "full": "boolean",
        "eps": "float",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.GroupNorm"] = {
    "args": {
        "num_groups": "integer",
        "num_channels": "integer"
    },
    "kwargs": {
        "eps": "float",
        "affine": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Hardshrink"] = {
    "args": {
        "lambd": "float"
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
    "args": {
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Hardswish"] = {
    "args": {
        "inplace": "boolean"
    },
    "kwargs": {},
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
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.HingeEmbeddingLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.HuberLoss"] = {
    "args": {
        "reduction": "string",
        "delta": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Identity"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.InstanceNorm1d_1"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.InstanceNorm2d_1"] = {
    "args": {
        "num_features": "integer",
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.InstanceNorm3d_1"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.InstanceNorm3d_2"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "device": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.KLDivLoss_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string",
        "log_target": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.KLDivLoss_2"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string",
        "log_target": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.L1Loss_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.L1Loss_2"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.LPPool1d"] = {
    "args": {
        "norm_type": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.LPPool2d_1"] = {
    "args": {
        "norm_type": "float",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.LPPool2d_2"] = {
    "args": {
        "norm_type": "float",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.LSTM"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
        "num_layers": "integer"
    },
    "kwargs": {
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "proj_size": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "h_0": "tensor",
            "c_0": "tensor"
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
        "bias": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "h_0": "tensor",
            "c_0": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LayerNorm"] = {
    "args": {
        "normalized_shape": "integer",
        "eps": "float",
        "elementwise_affine": "boolean",
        "bias": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LazyBatchNorm1d_1"] = {
    "args": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LazyBatchNorm2d"] = {
    "args": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LazyInstanceNorm1d_1"] = {
    "args": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LazyInstanceNorm2d_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.LeakyReLU"] = {
    "args": {
        "negative_slope": "float",
        "inplace": "boolean"
    },
    "kwargs": {},
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
        "bias": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LocalResponseNorm"] = {
    "args": {
        "size": "integer",
        "alpha": "float",
        "beta": "float",
        "k": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LogSigmoid"] = {
    "args": {
        "input": "tensor"
    },
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
        "dim": "integer"
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
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
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
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.MaxPool1d"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.MaxPool2d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.MaxPool3d_1"] = {
    "args": {
        "kernel_size": "integer",
        "stride": "integer"
    },
    "kwargs": {
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.MaxPool3d_2"] = {
    "args": {
        "kernel_size": "tuple",
        "stride": "tuple"
    },
    "kwargs": {
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.MaxUnpool1d"] = {
    "args": {
        "kernel_size": "integer",
        "stride": "integer"
    },
    "kwargs": {
        "padding": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "integer"
        }
    }
}
signatures["torch.nn.MaxUnpool2d"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "integer"
        }
    }
}
signatures["torch.nn.MaxUnpool3d"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "output_size": "integer"
    },
    "inner": {}
}
signatures["torch.nn.Mish"] = {
    "args": {
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MultiLabelMarginLoss_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.MultiLabelSoftMarginLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.MultiMarginLoss"] = {
    "args": {
        "p": "integer",
        "margin": "float",
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.MultiheadAttention_1"] = {
    "args": {
        "embed_dim": "integer",
        "num_heads": "integer"
    },
    "kwargs": {
        "dropout": "float",
        "bias": "boolean",
        "add_bias_kv": "boolean",
        "add_zero_attn": "boolean",
        "kdim": "integer",
        "vdim": "integer",
        "batch_first": "boolean"
    },
    "inner": {
        "args": {
            "query": "tensor",
            "key": "tensor",
            "value": "tensor"
        },
        "kwargs": {
            "key_padding_mask": "tensor",
            "need_weights": "boolean",
            "attn_mask": "tensor",
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
        "size_average": "boolean",
        "ignore_index": "integer",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.PReLU_1"] = {
    "args": {
        "num_parameters": "integer",
        "init": "float"
    },
    "kwargs": {
        "device": "string",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["torch.nn.PReLU_2"] = {
    "args": {
        "num_parameters": "integer"
    },
    "kwargs": {
        "init": "float",
        "device": "string",
        "dtype": "dtype"
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
        "p": "float",
        "eps": "float",
        "keepdim": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input1": "tensor",
            "input2": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Parameter"] = {
    "args": {
        "data": "tensor",
        "requires_grad": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.ParameterList"] = {
    "args": {
        "values": "tensor_list"
    },
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
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.PixelUnshuffle_1"] = {
    "args": {
        "downscale_factor": "integer"
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
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "log_input": "boolean",
        "full": "boolean",
        "size_average": "boolean",
        "eps": "float",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.RMSNorm"] = {
    "args": {
        "normalized_shape": "integer"
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RNN"] = {
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
        "bidirectional": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hx": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RNNCell_1"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "bias": "boolean",
        "nonlinearity": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hidden": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RNNCell_2"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "bias": "boolean",
        "nonlinearity": "string",
        "device": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hidden": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RReLU"] = {
    "args": {
        "lower": "float",
        "upper": "float"
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
signatures["torch.nn.ReLU"] = {
    "args": {
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReLU6"] = {
    "args": {
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad1d_1"] = {
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
signatures["torch.nn.ReflectionPad1d_2"] = {
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
signatures["torch.nn.ReflectionPad2d_1"] = {
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

signatures["torch.nn.ReflectionPad2d_2"] = {
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
signatures["torch.nn.ReplicationPad1d_1"] = {
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
signatures["torch.nn.ReplicationPad1d_2"] = {
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
signatures["torch.nn.ReplicationPad2d"] = {
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

signatures["torch.nn.ReplicationPad2d_1"] = {
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
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.SiLU"] = {
    "args": {
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Sigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.SmoothL1Loss_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string",
        "beta": "float"
    },
    "inner": {}
}
signatures["torch.nn.SoftMarginLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.Softmax"] = {
    "args": {
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softmax2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softmin"] = {
    "args": {
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softplus_1"] = {
    "args": {
        "beta": "float",
        "threshold": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softshrink"] = {
    "args": {
        "lambd": "float"
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
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.SyncBatchNorm"] = {
    "args": {
        "num_features": "integer",
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "process_group": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.Tanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Tanhshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
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
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Transformer_1"] = {
    "args": {
        "d_model": "integer",
        "nhead": "integer",
        "num_encoder_layers": "integer",
        "num_decoder_layers": "integer",
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string",
        "custom_encoder": "tensor",
        "custom_decoder": "tensor",
        "layer_norm_eps": "float",
        "batch_first": "boolean",
        "norm_first": "boolean",
        "bias": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "src": "tensor",
            "tgt": "tensor"
        },
        "kwargs": {
            "src_mask": "tensor",
            "tgt_mask": "tensor",
            "memory_mask": "tensor",
            "src_key_padding_mask": "tensor",
            "tgt_key_padding_mask": "tensor",
            "memory_key_padding_mask": "tensor",
            "src_is_causal": "boolean",
            "tgt_is_causal": "boolean",
            "memory_is_causal": "boolean"
        }
    }
}
signatures["torch.nn.TransformerDecoderLayer_1"] = {
    "args": {
        "d_model": "integer",
        "nhead": "integer"
    },
    "kwargs": {
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string",
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
            "tgt_mask": "tensor",
            "memory_mask": "tensor",
            "tgt_key_padding_mask": "tensor",
            "memory_key_padding_mask": "tensor",
            "tgt_is_causal": "boolean",
            "memory_is_causal": "boolean"
        }
    }
}
signatures["torch.nn.TransformerEncoderLayer_1"] = {
    "args": {
        "d_model": "integer",
        "nhead": "integer"
    },
    "kwargs": {
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string",
        "layer_norm_eps": "float",
        "batch_first": "boolean",
        "norm_first": "boolean",
        "bias": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.TransformerEncoderLayer_2"] = {
    "args": {
        "d_model": "integer",
        "nhead": "integer"
    },
    "kwargs": {
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string",
        "layer_norm_eps": "float",
        "batch_first": "boolean",
        "norm_first": "boolean",
        "bias": "boolean"
    },
    "inner": {
        "args": {
            "src": "tensor"
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
        "margin": "float",
        "p": "integer",
        "eps": "float",
        "swap": "boolean"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "anchor": "tensor",
            "positive": "tensor",
            "negative": "tensor"
        },
        "kwargs": {}
    }
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
        "kernel_size": "tuple",
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.UninitializedBuffer"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.UninitializedParameter"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Upsample_1"] = {
    "args": {
        "scale_factor": "float"
    },
    "kwargs": {
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Upsample_2"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Upsample_3"] = {
    "args": {
        "scale_factor": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Upsample_4"] = {
    "args": {
        "size": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ZeroPad1d_1"] = {
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

signatures["torch.nn.ZeroPad1d_2"] = {
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
signatures["torch.nn.ZeroPad2d_1"] = {
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
        "input": "tensor"
    },
    "kwargs": {
        "output_size": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.adaptive_avg_pool2d_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "output_size": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.adaptive_avg_pool2d_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "output_size": "tuple"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool3d_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "output_size": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.nn.functional.adaptive_max_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.adaptive_max_pool2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "output_size": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool3d_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "output_size": "integer",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.affine_grid"] = {
    "args": {
        "theta": "tensor",
        "size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.alpha_dropout"] = {
    "args": {
        "input": "tensor",
        "alpha": "float"
    },
    "kwargs": {
        "training": "boolean",
        "inplace": "boolean"
    },
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
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.avg_pool2d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.avg_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
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
        "bias": "tensor"
    },
    "kwargs": {
        "training": "boolean",
        " momentum": "float",
        " eps": "float",
        " cudnn_enabled": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.bilinear"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.binary_cross_entropy"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.binary_cross_entropy_with_logits"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        " reduction": "string",
        "pos_weight": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.celu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv1d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.conv2d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.conv_transpose1d_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "integer"
    },
    "inner": {}
}

signatures["torch.nn.functional.conv_transpose1d_2"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {}
}
signatures["torch.nn.functional.conv_transpose2d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.conv_transpose3d_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.cosine_embedding_loss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.cosine_similarity"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        " eps": "float",
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.cross_entropy_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "ignore_index": "integer",
        "reduction": "string",
        "label_smoothing": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.ctc_loss"] = {
    "args": {
        "log_probs": "tensor",
        "targets": "tensor_list",
        "input_lengths": "integer",
        "target_lengths": "integer"
    },
    "kwargs": {
        "blank": "integer",
        "reduction": "string",
        "zero_infinity": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.dropout"] = {
    "args": {
        "input": "tensor",
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.dropout2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.dropout3d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.elu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.embedding_1"] = {
    "args": {
        "input": "tensor",
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
        "input": "tensor",
        "embedding_dim": "integer",
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float"
    },
    "kwargs": {
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean",
        "per_sample_weights": "tensor",
        "include_last_offset": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.feature_alpha_dropout"] = {
    "args": {
        "input": "tensor",
        "alpha": "float"
    },
    "kwargs": {
        "training": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.fold_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple",
        "kernel_size": "integer"
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.fractional_max_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        " stride": "integer",
        " padding": "integer",
        " dilation": "integer",
        " ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.gelu_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "approximate": "string"
    },
    "inner": {}
}
signatures["torch.nn.functional.glu"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.grid_sample"] = {
    "args": {
        "input": "tensor",
        "grid": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "padding_mode": "string",
        "align_corners": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.group_norm"] = {
    "args": {
        "input": "tensor",
        "num_groups": "integer",
        "num_channels": "integer"
    },
    "kwargs": {
        "eps": "float",
        "affine": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.gumbel_softmax_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "tau": "float",
        "hard": "boolean",
        "dim": "integer",
        "temperature": "float"
    },
    "inner": {}
}
signatures["torch.nn.functional.hardshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float"
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
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.hardtanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min_val": "float",
        "max_val": "float",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.hinge_embedding_loss"] = {
    "args": {
        "input": "tensor",
        "embedding": "tensor"
    },
    "kwargs": {
        "margin": "float",
        " reduction": "string",
        "l2_reg": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.huber_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        " reduction": "string",
        "delta": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.instance_norm"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor",
        "running_mean": "tensor",
        "running_var": "tensor"
    },
    "kwargs": {
        " eps": "float",
        "affine": "boolean",
        "compute_stats": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.interpolate"] = {
    "args": {
        "input": "tensor",
        "size": "tuple",
        "scale_factor": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean",
        "antialias": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.kl_div_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        " reduction": "string",
        " log_target": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.l1_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        " reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.layer_norm"] = {
    "args": {
        "input": "tensor",
        "normalized_shape": "tuple",
        "weight": "tensor",
        "bias": "tensor",
        "eps": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.leaky_relu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "negative_slope": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.linear_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor"
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
        "scale": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.log_softmax_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["torch.nn.functional.log_softmax_2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "dtype": "dtype"
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
        "p": "float"
    },
    "kwargs": {
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.lp_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "p": "float"
    },
    "kwargs": {
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.margin_ranking_loss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        " reduction": "string",
        "size_average": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.max_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "ceil_mode": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.max_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        " dilation": "integer",
        " ceil_mode": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.max_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        " dilation": "tuple",
        " ceil_mode": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.max_pool3d_3"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        " dilation": "integer",
        " ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_unpool1d"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.max_unpool2d"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.max_unpool3d_1"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.nn.functional.max_unpool3d_2"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.mish"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.mse_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.multi_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string",
        "weight": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.multilabel_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.multilabel_soft_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string",
        "weight": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.functional.nll_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.functional.normalize"] = {
    "args": {
        "input": "tensor",
        "p": "float",
        "dim": "integer"
    },
    "kwargs": {
        "eps": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.one_hot"] = {
    "args": {
        "input": "tensor",
        "num_classes": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.pad"] = {
    "args": {
        "input": "tensor",
        "pad": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "value": "float"
    },
    "inner": {}
}
signatures["torch.nn.functional.pairwise_distance"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor"
    },
    "kwargs": {
        "p": "float",
        " eps": "float",
        " keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.pdist"] = {
    "args": {
        "input": "tensor",
        "p": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.pixel_shuffle"] = {
    "args": {
        "input": "tensor",
        "upsample_factor": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.pixel_unshuffle"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.poisson_nll_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        " reduction": "string",
        " log_prob": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.prelu"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.relu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.relu6"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.rrelu"] = {
    "args": {
        "input": "tensor",
        "lower": "float",
        "upper": "float"
    },
    "kwargs": {
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.selu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.silu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.smooth_l1_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        " reduction": "string",
        " beta": "float",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.soft_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        " reduction": "string",
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.softmax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
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
    "inner": {}
}
signatures["torch.nn.functional.softplus"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "beta": "integer",
        "threshold": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.softshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float"
    },
    "inner": {}
}
signatures["torch.nn.functional.softsign"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.tanhshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.threshold_1"] = {
    "args": {
        "input": "tensor",
        "threshold": "float",
        "value": "float"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.triplet_margin_loss"] = {
    "args": {
        "anchor": "tensor",
        "positive": "tensor",
        "negative": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "p": "integer",
        " eps": "float",
        " reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.unfold"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.calculate_gain"] = {
    "args": {
        "gain": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.constant__1"] = {
    "args": {
        "tensor": "tensor",
        "val": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.dirac__1"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.eye__"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.init.kaiming_normal__1"] = {
    "args": {
        "tensor": "tensor",
        "a": "float"
    },
    "kwargs": {
        "mode": "string",
        "nonlinearity": "string"
    },
    "inner": {}
}
signatures["torch.nn.init.kaiming_uniform_"] = {
    "args": {
        "tensor": "tensor",
        "a": "float"
    },
    "kwargs": {
        "mode": "string",
        "nonlinearity": "string"
    },
    "inner": {}
}
signatures["torch.nn.init.normal__1"] = {
    "args": {
        "tensor": "tensor",
        "mean": "float",
        "std": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.ones__1"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.orthogonal__1"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "gain": "float",
        "a": "integer"
    },
    "inner": {},
}

signatures["torch.nn.init.orthogonal__2"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "gain": "float",
        "a": "integer"
    },
    "inner": {},
}
signatures["torch.nn.init.sparse_"] = {
    "args": {
        "tensor": "tensor",
        "density": "float"
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
        "generator": "tensor_list",
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.init.xavier_normal__1"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "gain": "float",
        "a": "float"
    },
    "inner": {},
}
signatures["torch.nn.init.xavier_uniform__1"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "gain": "float",
        "a": "float"
    },
    "inner": {},
}
signatures["torch.nn.init.zeros_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "other": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.utils.clip_grad_norm__1"] = {
    "args": {
        "parameters": "tensor_list"
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "string"
    },
    "inner": {}
}
signatures["torch.nn.utils.clip_grad_value__"] = {
    "args": {
        "parameters": "tensor_list"
    },
    "kwargs": {
        "clip_value": "float"
    },
    "inner": {},
}
signatures["torch.nn.utils.get_total_norm"] = {
    "args": {
        "parameters": "tensor_list"
    },
    "kwargs": {
        "norm_type": "float",
        "error_if_nonfinite": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.utils.parameters_to_vector"] = {
    "args": {
        "parameters": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.utils.remove_weight_norm"] = {
    "args": {
        "module": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {},
}
signatures["torch.nn.utils.rnn.pack_padded_sequence"] = {
    "args": {
        "input": "tensor",
        "lengths": "tensor_list"
    },
    "kwargs": {
        "batch_first": "boolean",
        "enforce_sorted": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.utils.rnn.pack_sequence"] = {
    "args": {
        "sequences": "tensor_list"
    },
    "kwargs": {
        "batch_first": "boolean",
        "padding_value": "float",
        "lengths": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.utils.rnn.pad_sequence"] = {
    "args": {
        "sequences": "tensor_list"
    },
    "kwargs": {
        "batch_first": "boolean",
        "padding_value": "float",
        "total_length": "integer"
    },
    "inner": {},
}
signatures["torch.nn.utils.vector_to_parameters"] = {
    "args": {
        "parameters": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nonzero_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor",
        "as_tuple": "boolean"
    },
    "inner": {},
}
signatures["torch.nonzero_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor",
        "as_tuple": "boolean"
    },
    "inner": {},
}
signatures["torch.norm_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "string",
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.norm_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.norm_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "string",
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.norm_4"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.normal"] = {
    "args": {
        "mean": "tensor",
        "std": "tensor"
    },
    "kwargs": {
        "generator": "tensor",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.normal_1"] = {
    "args": {
        "mean": "float",
        "std": "tensor"
    },
    "kwargs": {
        "generator": "tensor",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.normal_2"] = {
    "args": {
        "mean": "tensor",
        "std": "float"
    },
    "kwargs": {
        "generator": "tensor",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.normal_3"] = {
    "args": {
        "mean": "float",
        "std": "float"
    },
    "kwargs": {
        "generator": "tensor",
        "out": "tensor",
        "size": "tuple"
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
signatures["torch.ones_like_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.outer"] = {
    "args": {
        "input": "tensor",
        "vec2": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.parse_schema"] = {
    "args": {
        "schema": "string"
    },
    "kwargs": {},
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
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "eps": "float"
    },
    "inner": {},
}
signatures["torch.permute"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.pinverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "rcond": "float"
    },
    "inner": {}
}
signatures["torch.polar"] = {
    "args": {
        "abs": "tensor",
        "angle": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.polygamma_1"] = {
    "args": {
        "n": "integer",
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.positive_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.pow_1"] = {
    "args": {
        "input": "tensor",
        "exponent": "float"
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
    "args": {
        "start_method": "string"
    },
    "kwargs": {},
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
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.put"] = {
    "args": {
        "input": "tensor",
        "index": "tensor"
    },
    "kwargs": {
        "value": "tensor",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.qr_1"] = {
    "args": {
        "input": "tensor",
        "some": "boolean"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.qr_2"] = {
    "args": {
        "input": "tensor",
        "some": "string"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.quantile"] = {
    "args": {
        "input": "tensor",
        "q": "float"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "interpolation": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.quantize_per_channel"] = {
    "args": {
        "input": "tensor",
        " scales": "tensor",
        " zero_points": "tensor",
        " axis": "integer",
        " dtype": "dtype"
    },
    "kwargs": {
        "q_scheme": "string"
    },
    "inner": {},
}
signatures["torch.quantize_per_tensor"] = {
    "args": {
        "input": "tensor",
        "scale": "float",
        "zero_point": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.rad2deg"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.rand_like_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "pin_memory": "boolean"
    },
    "inner": {},
}

signatures["torch.rand_like_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "pin_memory": "boolean",
        "memory_format": "string"
    },
    "inner": {},
}
signatures["torch.randint_like"] = {
    "args": {
        "input": "tensor",
        "low": "integer",
        "high": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.randn_like_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.range"] = {
    "args": {
        "start": "float",
        "end": "float"
    },
    "kwargs": {
        "step": "float",
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.ravel_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.real_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.reciprocal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.reciprocal_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.relu__"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
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
    "inner": {},
}
signatures["torch.repeat_interleave"] = {
    "args": {
        "input": "tensor",
        "repeats": "integer",
        "dim": "integer"
    },
    "kwargs": {
        "output": "tensor"
    },
    "inner": {},
}
signatures["torch.reshape"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.result_type"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.roll_1"] = {
    "args": {
        "input": "tensor",
        "shifts": "integer"
    },
    "kwargs": {
        "dims": "integer"
    },
    "inner": {}
}

signatures["torch.roll_2"] = {
    "args": {
        "input": "tensor",
        "shifts": "tuple"
    },
    "kwargs": {
        "dims": "tuple"
    },
    "inner": {}
}
signatures["torch.rot90_1"] = {
    "args": {
        "input": "tensor",
        "k": "integer",
        "dims": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.round"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "decimals": "integer",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.row_stack"] = {
    "args": {
        " tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.rsqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": " tensor"
    },
    "inner": {},
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
    "kwargs": {},
    "inner": {},
}
signatures["torch.scatter_1"] = {
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
    "kwargs": {},
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
        "reduce": "string",
        "include_self": "boolean"
    },
    "inner": {},
}
signatures["torch.searchsorted"] = {
    "args": {
        "sorted_sequence": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "out_int32": "boolean",
        "right": "boolean",
        "side": "string",
        "out": "tensor",
        "sorter": "tensor"
    },
    "inner": {}
}
signatures["torch.select_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "integer"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.select_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.select_copy"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_anomaly_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_cache_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_cpu_dtype"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_cpu_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_ipu_dtype"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_ipu_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.set_autocast_xla_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.set_default_device"] = {
    "args": {
        "device": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.set_default_dtype"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.set_default_tensor_type"] = {
    "args": {
        "tensor_type": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_deterministic_debug_mode"] = {
    "args": {
        "mode": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_flush_denormal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_grad_enabled"] = {
    "args": {
        "mode": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.set_num_interop_threads"] = {
    "args": {
        "num_interop_threads": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_num_threads"] = {
    "args": {
        "num_threads": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.set_warn_always"] = {
    "args": {
        "always": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.sgn"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.sigmoid_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sign"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.signbit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sin_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
    "inner": {},
}
signatures["torch.sort"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "descending": "boolean"
    },
    "kwargs": {
        "stable": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.sparse_bsr_tensor"] = {
    "args": {
        "input": "tensor",
        "nnz": "integer",
        "csr": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_coo_tensor"] = {
    "args": {
        "indices": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "size": "tuple",
        "dtype": "dtype",
        "layout": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_csr_tensor"] = {
    "args": {
        "crow": "tensor",
        "col": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "device": "string"
    },
    "inner": {},
}
signatures["torch.special.bessel_j1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.bessel_y0"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.bessel_y1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.digamma"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.entr"] = {
    "args": {
        "input": "tensor"
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
        "input": "tensor"
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
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
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
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.gammainc"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.gammaincc"] = {
    "args": {
        "input": "tensor",
        "a": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.gammaln"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.hermite_polynomial_he"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.i0"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.i0e"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.i1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.i1e"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.log1p"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.log_ndtr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.log_softmax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.special.logit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "eps": "float"
    },
    "inner": {},
}
signatures["torch.special.logsumexp"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.special.modified_bessel_i0"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
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
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.ndtr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.ndtri"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.polygamma"] = {
    "args": {
        "n": "integer",
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.psi"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.round"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
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
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.special.spherical_bessel_j0_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.xlog1py"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.xlogy"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.zeta"] = {
    "args": {
        "input": "tensor",
        "order": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.split_1"] = {
    "args": {
        "tensor": "tensor",
        "split_size_or_sections": "integer"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {}
}

signatures["torch.split_2"] = {
    "args": {
        "tensor": "tensor",
        "split_size_or_sections": "list"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {}
}
signatures["torch.spmm"] = {
    "args": {
        "input": "tensor",
        "matrices": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.sqrt__1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.square"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.square"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.squeeze_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.squeeze_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.squeeze_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.sspaddmm"] = {
    "args": {
        "input": "tensor",
        "mat1": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float"
    },
    "inner": {},
}
signatures["torch.stack"] = {
    "args": {
        "tensors": "tensor_list",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.std_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "correction": "integer",
        "keepdim": "boolean"
    },
    "inner": {}
}

signatures["torch.std_2"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "correction": "integer",
        "keepdim": "boolean"
    },
    "inner": {}
}

signatures["torch.std_3"] = {
    "args": {
        "input": "tensor",
        "dim": "None"
    },
    "kwargs": {
        "correction": "integer",
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.std_mean"] = {
    "args": {
        "input": "tensor",
        "unbiased": "boolean"
    },
    "kwargs": {
        "keepdim": "boolean",
        "correction": "integer"
    },
    "inner": {},
}
signatures["torch.stft"] = {
    "args": {
        "input": "tensor",
        "n_fft": "integer"
    },
    "kwargs": {
        "hop_length": "integer",
        "win_length": "integer",
        "window": "tensor",
        "center": "boolean",
        "pad_mode": "string",
        "normalized": "boolean",
        "onesided": "boolean",
        "return_complex": "boolean"
    },
    "inner": {}
}
signatures["torch.sub"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
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
        "out": "tensor"
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
    "inner": {},
}
signatures["torch.sum_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.sum_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.svd"] = {
    "args": {
        "input": "tensor",
        "some": "boolean",
        "compute_uv": "boolean"
    },
    "kwargs": {
        "out": "tuple"
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
    "inner": {},
}
signatures["torch.swapdims"] = {
    "args": {
        "input": "tensor",
        "dim0": "integer",
        "dim1": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.sym_float"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sym_fresh_size"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sym_int"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.t_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.take"] = {
    "args": {
        "input": "tensor",
        "index": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.tan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": " tensor"
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
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.tensordot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
        "dims": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.threshold"] = {
    "args": {
        "input": "tensor",
        "threshold": "float"
    },
    "kwargs": {
        "value": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.tile"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.topk"] = {
    "args": {
        "input": "tensor",
        "k": "integer"
    },
    "kwargs": {
        "dim": "integer",
        "largest": "boolean",
        "sorted": "boolean",
        "out": "tuple"
    },
    "inner": {}
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
signatures["torch.trapz_1"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {}
}
signatures["torch.triangular_solve"] = {
    "args": {
        "input": "tensor",
        "A": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "transpose": "boolean",
        "unitriangular": "boolean"
    },
    "inner": {},
}
signatures["torch.tril_1"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tril_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "diagonal": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tril_indices"] = {
    "args": {
        "input": "tensor",
        "k": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.triu"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.triu_indices"] = {
    "args": {
        "input": "tensor",
        "k": "integer"
    },
    "kwargs": {
        "diagonal": "integer"
    },
    "inner": {},
}
signatures["torch.true_divide"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.trunc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": " tensor"
    },
    "inner": {},
}
signatures["torch.trunc_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.typename"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unbind_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.unique_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "sorted": "boolean",
        "return_inverse": "boolean",
        "return_counts": "boolean",
        "dim": "integer"
    },
    "inner": {}
}
signatures["torch.unique_consecutive"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "as_tuple": "boolean"
    },
    "inner": {},
}
signatures["torch.unravel_index"] = {
    "args": {
        "indices": "tensor",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unsafe_split_with_sizes"] = {
    "args": {
        "input": "tensor",
        "split_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unsqueeze"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.vander"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "N": "integer",
        "increasing": "boolean"
    },
    "inner": {},
}
signatures["torch.var"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "correction": "integer",
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.var_mean"] = {
    "args": {
        "input": "tensor",
        "unbiased": "boolean"
    },
    "kwargs": {
        "keepdim": "boolean",
        "correction": "integer"
    },
    "inner": {},
}
signatures["torch.vdot"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.vsplit"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.vsplit_1"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.vstack"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.where_1"] = {
    "args": {
        "condition": "tensor",
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.where_2"] = {
    "args": {
        "condition": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.zeros"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",
        "device": "torch.device",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.zeros_like_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
