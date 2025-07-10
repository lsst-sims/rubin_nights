import io
import logging
import os
import pickle
import warnings

import numpy as np

from .connections import usdf_lfa

try:
    from lsst.resources import ResourcePath
except ModuleNotFoundError:
    warnings.warn("Install lsst.resources to use this module.")
    HAS_LSST_RESOURCES = False

try:
    import h5py
    import healpy as hp
except ModuleNotFoundError:
    warnings.warn("Install healpy and h5py to retrieve DREAM data.")

logger = logging.getLogger(__name__)


def get_scheduler_snapshot(uri: str, at_usdf: bool = True):
    if at_usdf:
        uri = usdf_lfa(uri, bucket="s3://lfa@")
        os.environ["LSST_DISABLE_BUCKET_VALIDATION"] = "1"
    resource = ResourcePath(uri).read()
    # unpickle
    scheduler, conditions, _ = pickle.loads(resource)
    return scheduler, conditions


def get_dream_cloud_maps(uri: str, at_usdf: bool = True) -> np.ndarray:
    if at_usdf:
        uri = usdf_lfa(uri, bucket="s3://lfa@")
        os.environ["LSST_DISABLE_BUCKET_VALIDATION"] = "1"
    resource = ResourcePath(uri)
    h5f = h5py.File(io.BytesIO(resource.read()), mode="r")
    cloud_map = h5f["clouds"][...]
    cloud_map = hp.reorder(cloud_map, n2r=True)
    return cloud_map
