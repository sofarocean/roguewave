_nopython = True
_nogil = True
_cache = True
_forceobj = False
_parallel = False
_error_model = "python"
_fastmath = False
_boundscheck = False

numba_default = {
    "nopython": _nopython,
    "nogil": _nogil,
    "cache": _cache,
    "forceobj": _forceobj,
    "parallel": _parallel,
    "error_model": _error_model,
    "fastmath": _fastmath,
    "boundscheck": _boundscheck,
}

numba_parallel = numba_default.copy()
numba_parallel["parallel"] = True

numba_nocache = numba_default.copy()
numba_nocache["cache"] = False

numba_nocache_parallel = numba_default.copy()
numba_nocache_parallel["cache"] = False
numba_nocache_parallel["parallel"] = True
