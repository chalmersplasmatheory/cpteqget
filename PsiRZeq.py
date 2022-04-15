# An equilibrium which only requires psi, R, Z (and rlim/zlim) as input.


from matplotlib._contour import QuadContourGenerator
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline, RectBivariateSpline


class PsiRZeq:
    

    def __init__(self, R, Z, psi, R0, Z0, psi0, psia, rlim, zlim, G, G_psi_n):
        """
        Constructor.

        :param R:       Major radius grid on which psi is given (1D).
        :param Z:       Vertical grid on which psi is given (1D).
        :param psi:     Poloidal flux as a function of (R,Z).
        :param psi0:    Poloidal flux at magnetic axis.
        :param psia:    Poloidal flux at plasma boundary.
        :param rlim:    Radial coordinates of limiting surface (e.g. wall).
        :param zlim:    Vertical coordinates of limiting surface (e.g. wall).
        :param G:       Toroidal magnetic field function (Btor = G/R).
        :param G_psi_n: Normalized poloidal flux grid for G function.
        """
        self._R = R
        self._Z = Z
        self._psi = psi
        self._G = G
        self._G_psi_n = G_psi_n

        self.R0   = R0
        self.Z0   = Z0
        self.psi0 = psi0
        self.psia = psia

        self.rlim = rlim
        self.zlim = zlim

        self.initialize_splines()


    def initialize_splines(self):
        """
        Initialize splines which are used to provide continuous versions
        of various input quantities.
        """
        self.psi  = RectBivariateSpline(self._R, self._Z, self._psi)
        self.G = InterpolatedUnivariateSpline(self._G_psi_n, self._G)

        # Set up contour generator
        psi2d  = self.psi(self._R, self._Z).T
        psin2d = (psi2d - self.psi0) / (self.psia-self.psi0)
        R, Z = np.meshgrid(self._R, self._Z)
        self.contour_generator = QuadContourGenerator(R, Z, psin2d, None, True, 0)

        psi_n = np.linspace(0, 1, self._R.size)

        r = np.zeros(psi_n.shape)
        R_major = np.zeros(psi_n.shape)

        for i, i_psiN in enumerate(psi_n[1:]):
            surface_R, surface_Z = self.get_flux_surface(psi_n=i_psiN)
            R_major[i+1] = self._R_midplane(surface_R, surface_Z)

        self.lcfs_R = surface_R
        self.lcfs_Z = surface_Z

        R_major[0] = R_major[1] + psi_n[1] * (R_major[2]-R_major[1]) / (psi_n[2]-psi_n[1])

        self.R_major = InterpolatedUnivariateSpline(psi_n, R_major)

        if self.R0 is None:
            self.R0 = self.R_major(0)

        for i in range(1, psi_n.size):
            r[i] = self._r_midplane(Rmid=R_major[i])

        self.a_minor = r[-1]
        self.r = InterpolatedUnivariateSpline(psi_n, r)
        self.rho = InterpolatedUnivariateSpline(psi_n, r/r[-1])


    def get_flux_surface(self, psi_n, theta=None):
        """
        Trace the flux surface for the given normalized psi.
        """
        vertices = self.contour_generator.create_contour(psi_n)
        
        # Find the contour which is "the most" closed
        # (since this is probably the only closed flux surface we're
        # looking for)
        mostClosed, cv = None, self.R0 if self.R0 is not None else 1

        for i in range(len(vertices)):
            if vertices[i].ndim == 2:
                closevalue = np.sqrt((vertices[i][0,0]-vertices[i][-1,0])**2 + (vertices[i][0,1]-vertices[i][-1,1])**2)

                if closevalue < cv:
                    mostClosed = i
                    cv = closevalue

        R, Z = vertices[mostClosed][:,0], vertices[mostClosed][:,1]

        if theta is not None:
            _theta = np.arctan2(R-self.R0, Z-self.Z0)
            #_theta[-1] = _theta[0] + 2*np.pi
            i = -1
            while _theta[i] < 0:
                _theta[i] += 2*np.pi
                i -= 1

            for i in range(1, _theta.size):
                # The contour finding routine may sometimes give us the
                # same point multiple times, so we have to remove them
                # manually...
                if _theta[i] == _theta[i-1]:
                    _tt = np.zeros((_theta.size-1,))
                    _tt[:i] = _theta[:i]
                    _tt[i:] = _theta[(i+1):]

                    _r = np.zeros((_theta.size-1,))
                    _r[:i] = R[:i]
                    _r[i:] = R[(i+1):]

                    _z = np.zeros((_theta.size-1,))
                    _z[:i] = Z[:i]
                    _z[i:] = Z[(i+1):]

                    _theta = _tt
                    R = _r
                    Z = _z
                    break

            _R = CubicSpline(_theta, R, bc_type='periodic')
            _Z = CubicSpline(_theta, Z, bc_type='periodic')

            R = _R(theta+np.pi/2)
            Z = _Z(theta+np.pi/2)

        return R, Z


    def parametrize_equilibrium(self, psi_n=None, npsi=40):
        """
        Calculates the magnetic equilibrum parameters used by the
        analytical magnetic field in DREAM for a range of flux surfaces.

        :param psi_n: List of normalized poloidal flux for which to calculate the parameters.
        :param npsi:  Number of psi points to calculate parameters for (uniformly spaced between (0, 1]).
        """
        if psi_n is None:
            psi_n = np.linspace(0, 1, npsi+1)[1:]

        radius, psi, kappa, delta, Delta, GOverR0, R, Z = [], [], [], [], [], [], [], []
        for p in psi_n:
            params = self._get_eq_parameters(p)

            radius.append(params['r_minor'])
            psi.append(params['psi'])
            kappa.append(params['kappa'])
            delta.append(params['delta'])
            Delta.append(params['Delta'])
            GOverR0.append(params['GOverR0'])
            R.append(params['R'])
            Z.append(params['Z'])

        radius = np.array(radius)
        psi = np.array(psi) * 2*np.pi / self.R0
        kappa = np.array(kappa)
        delta = np.array(delta)
        Delta = np.array(Delta)
        GOverR0 = np.array(GOverR0)

        return {
            'R0': self.R0,
            'r': radius,
            'psi': psi,
            'kappa': kappa,
            'delta': delta,
            'Delta': Delta,
            'GOverR0': GOverR0
        }


    def _get_eq_parameters(self, psi_n):
        """
        Calculates the magnetic equilibrium parameters used by the
        analytical magnetic field in DREAM for a *SINGLE* flux surface.
        """
        R, Z = self.get_flux_surface(psi_n)

        rho = self.rho(psi_n)
        r_minor = rho * self.a_minor
        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]
        drho_dpsi = self.rho.derivative()(psi_n)
        R0 = self.R0
        R_major = self.R_major(psi_n)

        # Shaping parameters
        psi = self.psi(R[0], Z[0])[0,0]
        kappa = (max(Z)-min(Z)) / (2*r_minor)
        delta = (R_major - R_upper) / r_minor
        #Delta = self.R_major.derivative()(psi_n) / drho_dpsi / self.a_minor
        Delta = (max(R)+min(R))/2 - R0
        GOverR0 = self.G(psi_n) / R0

        return {
            'r_minor': r_minor,
            'psi': psi,
            'kappa': kappa,
            'delta': delta,
            'Delta': Delta,
            'GOverR0': GOverR0,
            'R': R,
            'Z': Z
        }


    def get_Br(self, R, Z):
        """
        Return the radial magnetic field component on the given (R,Z) grid.
        """
        Br = 1/R * self.psi(R, Z, dy=1, grid=False)
        return Br


    def get_Bz(self, R, Z):
        """
        Return the vertical magnetic field component on the given (R,Z) grid.
        """
        Bz = 1/R * self.psi(R, Z, dx=1, grid=False)
        return Bz


    def get_Btor(self, R, Z):
        """
        Return the toroidal magnetic field component on the given (R,Z) grid.
        """
        psi = self.psi(R, Z, grid=False)
        psi_n = (psi - self.psi0) / (self.psia - self.psi0)

        G  = self.G(psi_n)
        Btor = G / R

        return Btor


    def plot_flux_surfaces(self, ax=None, nr=10, ntheta=200, fit=True, *args, **kwargs):
        """
        Plot the flux surfaces of this magnetic equilibrium.

        :param ax:       Matplotlib Axes object to use for drawing.
        :param nr:       Number of flux surfaces to plot.
        :param ntheta:   Number of poloidal angles to plot (for contour fit).
        :param fit:      If ``True``, plots the DREAM parameter fit surfaces instead of the actual flux surfaces.
        :param *args:    Arguments for ``ax.plot()``.
        :param **kwargs: Keyword arguments for ``ax.plot()``.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if fit:
            theta = np.linspace(0, 2*np.pi, ntheta)
            p = self.parametrize_equilibrium(npsi=nr)

            for i in range(p['r'].size):
                R = p['R0'] + p['Delta'][i] + p['r'][i]*np.cos(theta + p['delta'][i]*np.sin(theta))
                Z = p['r'][i]*p['kappa'][i]*np.sin(theta)

                ax.plot(R, Z, *args, **kwargs)
            ax.axis('equal')
        else:
            psi_n = np.linspace(0, 1, nr+1)[1:]
            for p in psi_n:
                R, Z = self.get_flux_surface(p)
                ax.plot(R, Z, *args, **kwargs)
            ax.axis('equal')

        return ax


    def get_LUKE(self, npsi=80, ntheta=90):
        """
        Returns equilibrium data in the LUKE equilibrium format.
        """
        theta = np.linspace(0, 2*np.pi, ntheta)
        psi_n = np.linspace(0, 1, npsi+1)[1:]

        Rp, Zp = self.R0, self.Z0
        psi_apRp = 2*np.pi * self.psi(Rp+self.r(psi_n), Zp) * self.a_minor / Rp

        ptx = np.zeros((psi_n.size, ntheta))
        pty = np.zeros((psi_n.size, ntheta))
        for i in range(npsi):
            ptx[i,:], pty[i,:] = self.get_flux_surface(psi_n[i], theta=theta)

        ptBx = self.get_Br(ptx, pty)
        ptBy = self.get_Bz(ptx, pty)
        ptBPHI = self.get_Btor(ptx, pty)

        return {
            'id': 'GEQDSK data',
            'Rp': np.array([Rp]), 'Zp': np.array([Zp]),
            'psi_apRp': psi_apRp,
            'theta': theta,
            'ptx': ptx.T-Rp, 'pty': pty.T-Zp,
            'ptBx': ptBx.T, 'ptBy': ptBy.T, 'ptBPHI': ptBPHI.T
        }


    def get_SOFT(self, nr=80, nz=100):
        """
        Returns equilibrium data in the SOFT equilibrium format.
        """
        # Generate R/Z grid
        _R = np.linspace(np.amin(self.rlim), np.amax(self.rlim))
        _Z = np.linspace(np.amin(self.zlim), np.amax(self.zlim))

        R, Z = np.meshgrid(_R, _Z)

        Bphi = self.get_Btor(R, Z)
        Br   = self.get_Br(R, Z)
        Bz   = self.get_Bz(R, Z)

        maxis = np.array([self.R0, self.Z0])
        desc = 'cpteqget'
        name = 'cpteqget'

        separatrix = np.array([self.lcfs_R, self.lcfs_Z]).T
        wall = np.array([self.rlim, self.zlim]).T

        verBphi = Bphi[0,:]
        verBr   = Br[0,:]
        verBz   = Bz[0,:]

        return {
            'Bphi': Bphi, 'Br': Br, 'Bz': Bz,
            'desc': desc, 'name': name, 'maxis': maxis,
            'r': R, 'z': Z,
            'separatrix': separatrix, 'wall': wall,
            'verBphi': verBphi, 'verBr': verBr, 'verBz': verBz
        }


    def save_eq_parameters(self, filename, nr=40):
        """
        Save the DREAM analytical equilibrium parameters corresponding to this
        GEQDSK file to an HDF5 file named ``filename``.
        """
        params = self.parametrize_equilibrium(npsi=nr)

        with h5py.File(filename, 'w') as f:
            f['r'] = params['r']
            f['Delta'] = params['Delta']
            f['delta'] = params['delta']
            f['GOverR0'] = params['GOverR0']
            f['kappa'] = params['kappa']
            f['psi_p'] = params['psi']

    
    def save_LUKE(self, filename, npsi=80, ntheta=90):
        """
        Save this equilibrium in a LUKE compatible equilibrium file.
        """
        equil = self.get_LUKE(npsi=npsi, ntheta=ntheta)

        with h5py.File(filename, 'w') as f:
            g = f.create_group('equil')

            for key, val in equil.items():
                g[key] = val


    def save_SOFT(self, filename, nr=80, nz=100):
        """
        Save this equilibrium in a SOFT compatible equilibrium file.
        """
        equil = self.get_SOFT(nr=nr, nz=nz)
        
        with h5py.File(filename, 'w') as f:
            for key, val in equil.items():
                f[key] = val


    def _R_midplane(self, R, Z):
        """
        Given the (R,Z) coordinates for a flux surface, calculate the major
        radius in the outer midplane (Z=z_axis) for that surface.
        """
        # Find Z=0
        idx = []
        for i in range(Z.size):
            if (Z[i-1]-self.Z0)*(Z[i]-self.Z0) < 0:
                idx.append(i-1)
        
        # Each index in 'idx' gives a point such that Z=0 between [i, i+1].
        imax, Rmax = 0, 0
        for i in idx:
            if R[i] > Rmax:
                imax, Rmax = i, R[i]

        # Interpolate to find actual Rmax
        r1, r2 = R[imax], R[imax+1]
        z1, z2 = Z[imax], Z[imax+1]
        Rmax = (r2-r1)/(z2-z1)*(0-z1) + r1

        return Rmax


    def _r_midplane(self, R=0, Z=0, Rmid=None):
        """
        Given the (R,Z) coordinates for a flux surface, calculate the minor
        radius in the outer midplane (Z=z_axis) for that surface.
        """
        if Rmid is None:
            if R == 0 or Z == 0:
                raise Exception("Invalid call to '_rho_midplane()'.")

            Rmid = self._R_midplane(R, Z)

        return (Rmid - self.R0)


