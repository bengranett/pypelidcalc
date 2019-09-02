"""Summary
"""
cimport cython_gsl as gsl


cdef class RNG(object):
    """"""
    def __cinit__(self, *args, **kwargs):
        self._rng = gsl.gsl_rng_alloc(gsl.gsl_rng_taus)

    def __deallocate__(self):
        gsl.gsl_rng_free(self._rng)

    cdef void seed(self, unsigned long int seed) nogil:
        """ Set the seed of the random number generator

        Parameters
        ----------
        seed : int
        """
        cdef gsl.gsl_rng * rng = self._rng
        gsl.gsl_rng_set(rng, seed)

    cdef double uniform(self) nogil:
        """ Draw a sample from the uniform distribution 0 <= x < 1

        Returns
        -------
        double : sample from uniform distribution
        """
        cdef gsl.gsl_rng * rng = self._rng
        return gsl.gsl_rng_uniform(rng)

    cdef double gaussian(self, double sigma) nogil:
        """ Draw a sample from the Gaussian distribution with width sigma.

        Parameters
        ----------
        sigma : float
            Standard deviation of Gaussian PDF

        Returns
        -------
        double
        """
        cdef gsl.gsl_rng * rng = self._rng
        return gsl.gsl_ran_gaussian(rng, sigma)

    cdef unsigned int poisson(self, double n) nogil:
        """ Draw a sample from the Poisson distribution with mean n.

        Parameters
        ----------
        n : float
            mean of distribution

        Returns
        -------
        double
        """
        cdef gsl.gsl_rng * rng = self._rng
        return gsl.gsl_ran_poisson(rng, n)


cpdef RNG rng = RNG()


def seed(unsigned long int seed):
    """ Set the seed of the random number generator

    Parameters
    ----------
    seed : int
    """
    rng.seed(seed)


def uniform():
    """ Draw a sample from the uniform distribution 0 <= x < 1

    Returns
    -------
    double : sample from uniform distribution
    """
    return rng.uniform()


def gaussian(double sigma):
    """ Draw a sample from the Gaussian distribution with width sigma.

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian PDF

    Returns
    -------
    double
    """
    return rng.gaussian(sigma)


def poisson(double n):
    """ Draw a sample from the Gaussian distribution with width sigma.

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian PDF

    Returns
    -------
    int
    """
    return rng.poisson(n)

