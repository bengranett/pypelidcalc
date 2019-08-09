import numpy as np
cimport numpy as np

from cython.view cimport array as cvarray
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cimport bulgy_disk


cdef double [:,:] sample_galaxy(galaxy_struct * gal, int n, double plate_scale=1.0):
    """ Random sample the galaxy profile.

    Parameters
    ----------
    gal : galaxy_struct
        galaxy data structure (c pointer)
    n : int
        number of samples to draw
    plate_scale : float
        conversion in arcsec/pixel to sample in pixel coordinates.

    Returns
    -------
    xy : n.ndarray
       Array of ixel coordinates with shape (n, 2)
    """
    cdef double bulge_scale_pix = gal.bulge_scale / plate_scale
    cdef double disk_scale_pix = gal.disk_scale / plate_scale

    return bulgy_disk.bulgy_disk_sample(bulge_scale_pix, disk_scale_pix, gal.bulge_fraction, gal.axis_ratio, gal.pa, n)


cdef class Galaxy:
    """ """

    def __init__(self, **params):
        """ """
        for key, value in params.items():
            self.__setattr__(key, value)

    def __cinit__(self):
        """ """
        cdef int i

        self.gal = <galaxy_struct *> PyMem_Malloc(sizeof(galaxy_struct))
        if not self.gal:
            raise MemoryError

        self.gal.line_count = 0
        self.gal.continuum_mag_1 = 0
        self.gal.continuum_mag_2 = 0
        self.gal.continuum_wavelength_1 = 0
        self.gal.continuum_wavelength_2 = 0
        self.gal.z = 0
        self.gal.disk_scale = 0
        self.gal.bulge_fraction = 0
        self.gal.axis_ratio = 1
        self.gal.pa = 0
        self.gal.offx = 0
        self.gal.offy = 0
        self.gal.velocity_disp = 0

        self.gal.emission_line = <el_struct *> PyMem_Malloc(LINE_COUNT_ALLOC*sizeof(el_struct))
        if not self.gal.emission_line:
            raise MemoryError
        for i in range(LINE_COUNT_ALLOC):
            e = self.gal.emission_line[i]
            e.wavelength = 0
            e.wavelength_obs = 0
            e.flux = 0
            e.variance = 0
            e.background = 0


    def __dealloc__(self):
        """ """
        PyMem_Free(self.gal.emission_line)
        PyMem_Free(self.gal)

    def __getattr__(self, name):
        """ """
        if name == 'z':
            return self.gal.z
        elif name == 'bulge_scale':
            return self.gal.bulge_scale
        elif name == 'disk_scale':
            return self.gal.disk_scale
        elif name == 'bulge_fraction':
            return self.gal.bulge_fraction
        elif name == 'axis_ratio':
            return self.gal.axis_ratio
        elif name == 'pa':
            return self.gal.pa
        elif name == 'offx':
            return self.gal.offx
        elif name == 'offy':
            return self.gal.offy
        elif name == 'velocity_disp':
            return self.gal.velocity_disp
        elif name == 'line_count':
            return self.gal.line_count
        elif name == 'continuum_mag_1':
            return self.gal.continuum_mag_1
        elif name == 'continuum_mag_2':
            return self.gal.continuum_mag_2
        elif name == 'continuum_wavelength_1':
            return self.gal.continuum_wavelength_1
        elif name == 'continuum_wavelength_2':
            return self.gal.continuum_wavelength_2
        elif name == 'emission_line':
            return self.get_emission_lines()
        else:
            raise AttributeError("Unkown name %s"%name)

    def __setattr__(self, name, value):
        """ """
        if name == 'z':
            self.gal.z = value
        elif name == 'bulge_scale':
            self.gal.bulge_scale = value
        elif name == 'disk_scale':
            self.gal.disk_scale = value
        elif name == 'bulge_fraction':
            self.gal.bulge_fraction = value
        elif name == 'axis_ratio':
            self.gal.axis_ratio = value
        elif name == 'pa':
            self.gal.pa = value
        elif name == 'offx':
            self.gal.offx = value
        elif name == 'offy':
            self.gal.offy = value
        elif name == 'velocity_disp':
            self.gal.velocity_disp = value
        elif name == 'continuum_mag_1':
            self.gal.continuum_mag_1 = value
        elif name == 'continuum_mag_2':
            self.gal.continuum_mag_2 = value
        elif name == 'continuum_wavelength_1':
            self.gal.continuum_wavelength_1 = value
        elif name == 'continuum_wavelength_2':
            self.gal.continuum_wavelength_2 = value
        elif name == 'emission_line':
            self.set_emission_lines(**value)
        elif name == 'line_count':
            self.gal.line_count = value
        else:
            raise AttributeError("Unkown name %s"%name)

    def __str__(self):
        """ """
        out = "< Galaxy\n"
        out += "  %i emission lines:\n"%self.gal.line_count
        for i in range(self.gal.line_count):
            out += "    %i) %gA:\t%g\t%g\t%g\n"%(i+1, self.gal.emission_line[i].wavelength, self.gal.emission_line[i].flux, self.gal.emission_line[i].variance, self.gal.emission_line[i].background)
        out += "  bulge_scale: %g\n"%self.gal.bulge_scale
        out += "  disk_scale: %g\n"%self.gal.disk_scale
        out += "  bulge_fraction: %g\n"%self.gal.bulge_fraction
        out += "  axis_ratio: %g\n"%self.gal.axis_ratio
        out += "  velocity_disp: %g\n"%self.gal.velocity_disp
        out += ">"
        return out

    def set(self, **params):
        """ """
        for key, value in params.items():
            self.__setattr__(key, value)

    def append_line(self, wavelength, wavelength_obs=None, flux=None, variance=None, background=None):
        """ """
        cdef int i, n, l, m, k, scalar
        cdef el_struct * new_list
        n = self.gal.line_count

        try:
            l = len(wavelength)
            scalar = 0
        except:
            l = 1
            scalar = 1

        if scalar == 1:
            if not flux: flux = 0
            if not variance: variance = 0
            if not background: background = 0
            if not wavelength_obs: wavelength_obs = 0
        else:
            if flux is None: flux = np.zeros(l, dtype='d')
            if variance is None: variance = np.zeros(l, dtype='d')
            if background is None: background = np.zeros(l, dtype='d')
            if wavelength_obs is None: wavelength_obs = np.zeros(l, dtype='d')

        m = n + l

        if m > LINE_COUNT_ALLOC:
            new_list =  <el_struct *> PyMem_Realloc(self.gal.emission_line, m*sizeof(el_struct))
            if not new_list:
                raise MemoryError
        else:
            new_list = self.gal.emission_line

        if scalar == 1:
            new_list[n].wavelength = wavelength
            new_list[n].wavelength_obs = wavelength_obs
            new_list[n].flux = flux
            new_list[n].variance = variance
            new_list[n].background = background
        else:
            for i in range(l):
                k = n + i
                new_list[k].wavelength = wavelength[i]
                new_list[k].wavelength_obs = wavelength_obs[i]
                new_list[k].flux = flux[i]
                new_list[k].variance = variance[i]
                new_list[k].background = background[i]

        self.gal.emission_line = new_list

        self.gal.line_count = m

    def get_emission_lines(self):
        """ """
        el_list = {'wavelength':[], 'wavelength_obs':[],'flux':[], 'variance':[], 'background':[]}
        el = self.gal.emission_line
        for i in range(self.gal.line_count):
            el_list['wavelength'].append(el[i].wavelength)
            el_list['wavelength_obs'].append(el[i].wavelength_obs)
            el_list['flux'].append(el[i].flux)
            el_list['variance'].append(el[i].variance)
            el_list['background'].append(el[i].background)
        return el_list

    def set_emission_lines(self, wavelength=None, wavelength_obs=None, flux=None, variance=None, background=None):
        """ """
        self.gal.line_count = 0
        if wavelength is None:
            return

        self.append_line(wavelength, wavelength_obs, flux, variance, background)

    cdef galaxy_struct *  get_struct(self):
        """ """
        return self.gal

    cpdef void compute_obs_wavelengths(self, z):
        """ """
        cdef int i
        cdef el_struct * el = self.gal.emission_line

        for i in range(self.gal.line_count):
            el[i].wavelength_obs = (1 + z) * el[i].wavelength

    cpdef double [:,:] sample(self, int n, double plate_scale):
        """ """
        return sample_galaxy(self.gal, n, plate_scale)