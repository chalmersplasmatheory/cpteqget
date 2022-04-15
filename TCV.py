# Script for exporting a magnetic equilibrium for DREAM from TCV.
# Note: this script requires an installation of the MDSplus Python interface.

import h5py
import numpy as np
import eqhelpers
from . import TCVEquilibrium

try:
    from MDSplus import Connection
    conn = Connection('tcvdata.epfl.ch')

    AVAILABLE = True
except:
    AVAILABLE = False


def isAvailable():
    """
    Returns ``True`` if this module can be used to fetch equilibrium data
    on this system.
    """
    return AVAILABLE


def getLUKE(shot, time, *args, **kwargs):
    """
    Returns magnetic equilibrium data for the  given time of the specified
    AUG shot. If ``filename`` is provided, the data is also saved to the
    named LUKE equilibrium data file.

    :param shot: TCV shot to fetch equilibrium data for.
    :param time: Time to fetch equilibrium for.
    :param filename: Name of file to store data in.
    """
    global conn

    conn.openTree('tcv_shot', shot)

    mf = TCVEquilibrium(tcv=conn, time=time)
    luke = mf.get_LUKE(*args, **kwargs)

    conn.closeAllTrees()

    return luke


def getShaping(shot, time, *args, equil=None, **kwargs):
    """
    Calculates DREAM shaping parameters corresponding to the given
    magnetic equilibrium.
    """
    if equil is None:
        conn.openTree('tcv_shot', shot)
        mf = TCVEquilibrium(tcv=conn, time=time)
        params = mf.parametrize_equilibrium(*args, **kwargs)
        conn.closeAllTrees()

        return params
    else:
        return eqhelpers.parametrize_equilibrium(**equil)


