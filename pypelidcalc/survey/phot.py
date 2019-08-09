""" Routines for photometry """
import numpy as np
from pypelidcalc.utils import consts

def mag_to_flux(mag, wavelength=15000):
    """ Convert magnitude to flux (erg/s/cm^2/A)

    Parameters
    ----------
    mag : float
        Magnitude
    wavelength: float
        Wavelength (Angstrom)
    zp : float
        Magnitude zeropoint

    Returns
    -------
    float : flux in erg/s/cm^2/A
    """
    return 10**(-0.4*(mag + consts.ABzp)) * consts.c / wavelength**2

def flux_to_mag(f, wavelength=15000, zp=48.6):
    """ Compute magnitude from flux """
    return -2.5*np.log10(f * wavelength**2 / consts.c) - zp

def dnu_dlambda(wavelength):
    """ Conversion factor from f_nu to f_lambda.
    The formula is
    f_nu dnu = f_lambda dlambda
    f_lambda = f_nu * (dnu/dlambda)
    dnu/dlambda = c/lambda^2

    Inputs
    ------
    wavelength - reference wavelength in angstroms

    Outputs
    -------
    dnu/dlambda
    """
    return consts.c/wavelength**2


def flux_to_photon(flux, surface, wavelength):
    """ Convert flux to photon count.


    Inputs
    ------
    flux    - incident flux (erg/cm^2/s)
    surface - surface area (cm^2)
    wavelength  - wavelength (angstrom)

    Output
    ------
    photon count (1/sec)
    """
    return flux * surface / ((consts.planck * consts.c) / wavelength)

def photon_to_flux(photons, surface, wavelength):
    """ Convert photon count to flux.


    Inputs
    ------
    photons - photon count (1/sec)
    surface - surface area (cm^2)
    wavelength  - wavelength (angstrom)

    Output
    ------
    flux    - incident flux (erg/cm^2/s)

    """
    return photons * ((consts.planck * consts.c) / wavelength) / surface

def continuum(wavelength, mag1, wavelength1, mag2, wavelength2):
    """ Compute the continuum from two broadband magnitudes using linear interpolation.

    Two magnitudes should be specified.  The central wavelengths of the bandpasses
    must be in consts.bandpasses

    Parameters
    ----------
    wavelength : float
        wavelength (A)
    mags : float
        magnitudes specified as key-value pairs where key is a recognized bandpass
    """
    dx = wavelength2 - wavelength1
    dy = mag2 - mag1

    m_interp = (wavelength - wavelength1) * dy / dx + mag1

    return 10**(-0.4*(m_interp + consts.ABzp)) * consts.c / wavelength**2
