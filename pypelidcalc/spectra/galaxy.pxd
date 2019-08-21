cdef int LINE_COUNT_ALLOC = 8

ctypedef struct el_struct:
    double wavelength
    double wavelength_obs
    double flux
    double variance
    double background
    int rest_frame

ctypedef struct galaxy_struct:

    int line_count
    el_struct * emission_line

    double continuum_mag_1
    double continuum_mag_2
    double continuum_wavelength_1
    double continuum_wavelength_2

    double z
    double bulge_scale
    double disk_scale
    double bulge_fraction
    double axis_ratio
    double pa

    double offx
    double offy

    double velocity_disp


cdef class Galaxy:
    """ """
    cdef galaxy_struct * gal

    cdef galaxy_struct *  get_struct(self)
    cpdef double [:,:] sample(self, int n, double plate_scale)
    cpdef void compute_obs_wavelengths(self, z)