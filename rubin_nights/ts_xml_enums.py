# xml_enums isolates the use of ts-xml enums here
# if ts-xml is installed, use the proper imports
# if not, substitute hard-coded versions and issue warning

import enum
import logging
from itertools import chain

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["ScriptState", "CSCState", "SalIndexExtended", "apply_enum"]

try:
    from lsst.ts.xml.enums.Script import ScriptState
    from lsst.ts.xml.enums.ScriptQueue import SalIndex
    from lsst.ts.xml.sal_enums import State as CSCState

except ModuleNotFoundError:

    logger.warning("Could not import ts-xml; substituting copies but please install lsst-ts-xml.")
    # Could not import ts_xml so define the enums here
    # These are fairly stable; but better to import from ts-xml.

    class CSCState(enum.IntEnum):
        """CSC summaryState constants."""

        OFFLINE = 4
        STANDBY = 5
        DISABLED = 1
        ENABLED = 2
        FAULT = 3

    class ScriptState(enum.IntEnum):
        """ScriptState constants."""

        UNKNOWN = 0
        UNCONFIGURED = 1
        CONFIGURED = 2
        RUNNING = 3
        PAUSED = 4
        ENDING = 5
        STOPPING = 6
        FAILING = 7
        DONE = 8
        STOPPED = 9
        FAILED = 10
        CONFIGURE_FAILED = 11

    class SalIndex(enum.IntEnum):
        """Allowed SAL indices for the bin scripts.

        The CSC allows other positive values, as well,
        but those should only be used for unit testing.
        """

        MAIN_TEL = 1
        AUX_TEL = 2
        OCS = 3


class ScriptQueueExtensions(enum.IntEnum):
    """Add assigned `salIndex` values for other portions of the
    scriptqueue context feed.
    """

    ERRORS = 4
    EXP_SIMONYI = 5
    EXP_AUX = 6
    AUTOLOG_SIMONYI = 10
    AUTOLOG_AUX = 11
    AUTOLOG_OTHER = 12
    NARRATIVE_LOG_SIMONYI = 20
    NARRATIVE_LOG_AUX = 21
    NARRATIVE_LOG_OTHER = 22


SalIndexExtended = enum.IntEnum(
    "SalIndexExtended", [(i.name, i.value) for i in chain(SalIndex, ScriptQueueExtensions)]
)


def apply_enum(x: pd.Series, column: str, enumvals: ScriptState | CSCState) -> str:
    """Apply one of the ts-xml enums"""
    return enumvals(x[column]).name
