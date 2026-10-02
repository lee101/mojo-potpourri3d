# FFI plumbing shared by the solver wrappers.


# The sparse solver's entry points take buffers as Int addresses so that it can be called from C.
# Inside Mojo the same buffers are Pointers, so recover the address before calling.
def addr_of[T: AnyType](p: Pointer[T, AnyOrigin[mut=True]]) -> Int:
    return Int(p.unsafe_bitcast[UInt8]())
