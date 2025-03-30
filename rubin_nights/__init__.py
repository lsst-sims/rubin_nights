from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rubin_nights")
except PackageNotFoundError:
    # package is not installed
    pass
