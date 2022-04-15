# A class for working with TCV equilibria

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

from . PsiRZeq import PsiRZeq


class TCVEquilibrium(PsiRZeq):
    

    def __init__(self, tcv, time):
        """
        Constructor.
        """
        self.tcv = tcv
        self.source = 'LIUQE.M'

        super().__init__(**self.load(time=time))


    def load(self, time, nr=40):
        """
        Load TCV equilibrium data for the specified time.
        """
        psi = self.tdi('_x=\\results::psi')
        R   = self.tdi('dim_of(_x,0)')
        Z   = self.tdi('dim_of(_x,1)')
        t   = self.tdi('dim_of(_x,2)')

        psi0 = self.tdi('\\results::psi_axis')
        psia = self.tdi('\\results::surface_flux')

        rlim = self.tdi('static("r_v:in")')
        zlim = self.tdi('static("z_v:in")')
        R0   = self.tdi('\\results::r_axis')
        Z0   = self.tdi('\\results::z_axis')

        G    = self.tdi(f'tcv_eq("rbtor_rho","{self.source}")[*,$1]', time)
        G_r  = self.tdi(f'tcv_eq("rho","{self.source}")')

        i = np.argmin(abs(time-t))

        # Select appropriate time
        psia = psia[i]
        psi0 = psi0[i] + psia
        psi = psi[i,:] + psia
        R0 = R0[i]
        Z0 = Z0[i]

        return {
            'R': R, 'Z': Z, 'psi': psi.T,
            'R0': R0, 'Z0': Z0,
            'psi0': psi0, 'psia': psia,
            'rlim': rlim, 'zlim': zlim,
            'G': G, 'G_psi_n': G_r
        }


    def tdi(self, mdspath, *args, typecast=np.array):
        """
        Make a TDI call to the underlying MDSplus tree.
        """
        if hasattr(self.tcv, 'tdi'):
            return self.tcv.tdi(mdspath, *args, typecast=typecast)
        else:
            # Assume this is an mdsip connection
            return typecast(self.tcv.get(mdspath, *args))


