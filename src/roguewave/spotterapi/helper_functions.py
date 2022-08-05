from pysofar.sofar import SofarApi
from typing import List

# SofarAPI instance.
_API = None


# Helper functions
def get_spotter_ids(sofar_api: SofarApi = None) -> List[str]:
    """
    Get a list of Spotter ID's that are available through this account.

    :param sofar_api: valid SofarApi instance.
    :return: List of spotters available through this account.
    """
    if sofar_api is None:
        sofar_api = _get_sofar_api()
    return sofar_api.device_ids
# -----------------------------------------------------------------------------


def _get_sofar_api() -> SofarApi:
    """
    Gets a new sofar API object if requested. Returned object is essentially a
    Singleton class-> next calls will return the stored object instead of
    creating a new class. For module internal use only.

    :return: instantiated SofarApi object
    """
    global _API
    if _API is None:
        _API = SofarApi()
    return _API
# -----------------------------------------------------------------------------