from pypelidcalc.survey cimport psf
from galaxy cimport Galaxy
cimport pypelidcalc.cutils.interpolate as interpolate
cimport cython_gsl as gsl


cdef enum:
    POISSON_SAMPLE_LIMIT = 100


cdef class LineSimulator(object):
    """ Simulate emission line spectra.

    Parameters
    ----------
    optics : pypelidcalc.instrument.Optics
        Optics instance
    extraction_window : int
        the full extraction window is :math:`2 w + 1`
    """
    cdef public int nx
    cdef public int ny
    cdef public double lambda_min
    cdef public double lambda_max
    cdef public double dispersion
    cdef public double plate_scale
    cdef public psf.PSF_model PSF

    cdef double photon_shoot_limit, c_kms
    cdef public int npix, extraction_window, norm_window
    cdef public double extraction_norm
    cdef double [:] binsx
    cdef double [:] binsy
    cdef public double [:] extraction_weights
    cdef interpolate.interpolate_regular transmission_func
    cdef gsl.gsl_rng * rng

    cdef double [:] image
    cdef double [:] image_tmp
    cdef double [:] image_noisy
    cdef double [:] noise_image
    cdef public double[:] noise_template

    cdef void init_noise_template(self)
    cpdef void make_noise_spectrum(self, Galaxy gal, double scale, double[:] noise, int window=?)

    cdef int check_outside_extraction_window(self, double y) nogil
    cdef double get_extraction_weight(self, double y) nogil

    cpdef double render_line(self, Galaxy gal,
                        double x,
                        double y,
                        double velocity_disp,
                        long n,
                        double [:] image)

    cdef double compute_snr(self, double [:] signal, double [:] var) nogil

    cpdef sample_spectrum(self, Galaxy gal)

    cpdef sample_noise(self, Galaxy gal)

    cdef double wavelength_to_pixel(self, double wave) nogil
    cdef double pixel_to_wavelength(self, double pix) nogil
    cdef void sample_gaussian_noise(self, double [:] image, double [:] var, double [:] image_out) nogil
