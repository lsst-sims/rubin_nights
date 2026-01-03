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
SCIENCE_PROGRAMS = ["BLOCK-365", "BLOCK-407", "BLOCK-408", "BLOCK-416", "BLOCK-417", "BLOCK-419"]
