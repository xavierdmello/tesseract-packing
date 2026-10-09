"""ctypes wrapper around src/native/libkernel4d.dylib, operating in place on torch CPU float64 tensors."""
import ctypes, os
import torch

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "native", "libkernel4d.dylib"))
_dp = ctypes.POINTER(ctypes.c_double)
_lib.run_steps.argtypes = [ctypes.c_int, ctypes.c_int, _dp, _dp, _dp, _dp, _dp, _dp, ctypes.c_double,
                           ctypes.POINTER(_dp), ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_double, ctypes.c_int, _dp, _dp]
_lib.grad_one.argtypes = [ctypes.c_int, _dp, _dp, _dp, _dp, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, _dp, _dp, _dp, _dp]
_lib.grad_one.restype = ctypes.c_double


def ptr(t):
    assert t.dtype == torch.float64 and t.is_contiguous() and t.device.type == "cpu"
    return ctypes.cast(t.data_ptr(), _dp)


class NativeAdam:
    """Adam state + stepping for a kernel.Batch (D=4, float64 CPU)."""
    def __init__(self, m, lr, prune=True):
        self.m, self.lr, self.prune = m, lr, int(prune)
        self.state = [torch.zeros_like(x) for x in (m.c, m.c, m.p, m.p, m.W, m.W, m.Bh, m.Bh)]
        self.t = ctypes.c_long(0)
        self.pen = torch.zeros(m.B, dtype=torch.float64); self.mx = torch.zeros(m.B, dtype=torch.float64)

    def steps(self, k):
        m = self.m
        for x in (m.c, m.p, m.W, m.Bh, m.s, m.r): assert x.is_contiguous()
        arr = (_dp * 8)(*[ptr(x) for x in self.state])
        _lib.run_steps(m.B, m.n, ptr(m.c), ptr(m.p), ptr(m.W), ptr(m.Bh), ptr(m.s), ptr(m.r), m.margin,
                       arr, ctypes.byref(self.t), k, self.lr, self.prune, ptr(self.pen), ptr(self.mx))
        return self.pen, self.mx

    def reset(self, idx):
        for x in self.state: x[idx] = 0
