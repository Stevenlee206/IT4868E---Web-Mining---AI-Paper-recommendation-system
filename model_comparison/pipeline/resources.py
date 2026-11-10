"""Windows process-memory observations without an additional dependency."""
def memory_usage():
    import sys
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.windll.kernel32; kernel.GetCurrentProcess.restype=wintypes.HANDLE
        query=ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not query(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb): raise ctypes.WinError()
        return {'working_set_bytes':counters.WorkingSetSize,'process_peak_working_set_bytes':counters.PeakWorkingSetSize}
    import resource
    return {'process_peak_working_set_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
