import numpy as np

# Reference values

# Endpoints for services at different sites
API_ENDPOINTS = {
    "usdf": "https://usdf-rsp.slac.stanford.edu",
    "usdf-dev": "https://usdf-rsp-dev.slac.stanford.edu",
    "summit": "https://summit-lsp.lsst.codes",
    "base": "https://base-lsp.lsst.codes",
}

# Bad visit repo sources
BAD_VISITS_LSSTCAM = (
    "https://raw.githubusercontent.com/lsst-dm/excluded_visits/" "refs/heads/main/LSSTCam/bad.ecsv"
)
BAD_VISITS_LSSTCOMCAM = (
    "https://raw.githubusercontent.com/lsst-dm/excluded_visits/" "refs/heads/main/LSSTComCam/bad.ecsv"
)

# Current FBS science programs
SCIENCE_PROGRAMS = ("BLOCK-365", "BLOCK-407", "BLOCK-408", "BLOCK-416", "BLOCK-417", "BLOCK-419", "BLOCK-421")

# Approximate pixel scale
PLATESCALE = 0.2
# convert from SIGMA to FWHM
SIGMA_TO_FWHM: float = 2.0 * np.sqrt(2.0 * np.log(2.0))
FWHM_TO_SIGMA: float = 1 / SIGMA_TO_FWHM


# TODO make these time-dependent
# current estimate for zeropoint offsets to be able to evaluate clouds=0
ZEROPOINT_OFFSETS_LSSTCAM = {"u": 0.12, "g": 0.09, "r": 0.13, "i": 0.13, "z": 0.14, "y": 0.02}
# lsstcomcam offsets based on refcats at the time of processing
ZEROPOINT_OFFSETS_LSSTCOMCAM = {"u": 0.26, "g": -0.14, "r": -0.09, "i": -0.10, "z": -0.13, "y": -0.18}
# lsstcomcam offsets for DP1 are probably 0 although might be
ZEROPOINT_OFFSETS_DP1 = {"u": 0.03, "g": 0.01, "r": 0.00, "i": 0.00, "z": -0.00, "y": 0.01}

