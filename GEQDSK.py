# 
# Class for loading and working with a magnetic equilibrium stored in a
# GEQDSK file.
#
# Written by: Mathias Hoppe, 2022
#

import numpy as np
import re
from . PsiRZeq import PsiRZeq


class GEQDSK(PsiRZeq):
    

    def __init__(self, filename):
        """
        Constructor.
        """
        super().__init__(**self.load(filename))


    def _next_value(self, fh):
        """
        Load the next value from the text stream 'fh'.
        """
        pattern = re.compile(r"[ +\-]?\d+(?:\.\d+(?:[Ee][\+\-]\d\d)?)?")

        for line in fh:
            matches = pattern.findall(line)
            for m in matches:
                if "." in m:
                    yield float(m)
                else:
                    yield int(m)


    def load(self, filename):
        """
        Load data from the named GEQDSK file to this object.
        """
        data = self.load_geqdsk(filename)
        return self.process_data(data)


    def load_geqdsk(self, filename, cocos=1):
        """
        Load the named GEQDSK file.
        """
        with open(filename) as fh:
            header = fh.readline()
            words = header.split()
            if len(words) < 3:
                raise ValueError("Expected at least 3 numbers on first line")

            nx, ny = int(words[-2]), int(words[-1])
            
            data = {"nx": nx, "ny": ny}
            fields = ["rdim", "zdim", "rcentr", "rleft", "zmid", "rmagx",
                      "zmagx", "simagx", "sibdry", "bcentr", "cpasma", "simagx",
                      None, "rmagx", None, "zmagx", None, "sibdry", None, None]

            values = self._next_value(fh)
            
            for f in fields:
                val = next(values)
                if f:
                    data[f] = val

            def _read_1d(n):
                """
                Read a 1D array of length n from the GEQDSK file.
                """
                val = np.zeros(n)
                for i in range(n):
                    val[i] = next(values)

                return val


            def _read_2d(n, m):
                """
                Read a 2D (n,m) array in Fortran order
                """
                val = np.zeros((n, m))
                for j in range(m):
                    for i in range(n):
                        val[i, j] = next(values)

                return val


            data["fpol"] = _read_1d(nx)
            data["pres"] = _read_1d(nx)
            data["ffprime"] = _read_1d(nx)
            data["pprime"] = _read_1d(nx)

            data["psi"] = _read_2d(nx, ny)

            data["qpsi"] = _read_1d(nx)

            # Ensure that psi is divided by 2pi
            if cocos > 10:
                for var in ["psi", "simagx", "sibdry"]:
                    data[var] /= 2 * pi

            nbdry = next(values)
            nlim = next(values)

            if nbdry > 0:
                data["rbdry"] = np.zeros(nbdry)
                data["zbdry"] = np.zeros(nbdry)
                for i in range(nbdry):
                    data["rbdry"][i] = next(values)
                    data["zbdry"][i] = next(values)

            if nlim > 0:
                data["rlim"] = np.zeros(nlim)
                data["zlim"] = np.zeros(nlim)
                for i in range(nlim):
                    data["rlim"][i] = next(values)
                    data["zlim"][i] = next(values)

            return data


    def process_data(self, data):
        """
        Load data from the given GEQDSK dictionary.
        """
        nr = data['nx']
        nz = data['ny']

        psi = data['psi']
        psi0 = data['simagx']
        psia = data['sibdry']

        psi_n = np.linspace(0, 1, nr)

        G = data['fpol']
        G_psi_n = psi_n

        self.ff_prime = InterpolatedUnivariateSpline(psi_n, data["ffprime"])
        self.q        = InterpolatedUnivariateSpline(psi_n, data["qpsi"])
        self.pressure = InterpolatedUnivariateSpline(psi_n, data["pres"])
        self.p_prime  = self.pressure.derivative()

        Z0 = data['zmagx']
        R = np.linspace(data["rleft"], data["rleft"]+data["rdim"], nr)
        Z = np.linspace(data["zmid"]-data["zdim"]/2, data["zmid"]+data["zdim"]/2, self.nz)

        return {
            'R': R, 'Z': Z, 'psi': psi,
            'R0': None, 'Z0': Z0,
            'psi0': psi0, 'psia': psia,
            'rlim': rlim, 'zlim': zlim,
            'G': G, 'G_psi_n': G_psi_n
        }


