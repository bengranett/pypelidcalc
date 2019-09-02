import logging

import numpy as np
from . import phot, psf
from pypelidcalc.cutils import interpolate
from scipy import integrate
from scipy import interpolate as scipy_interpolate

def load_transmission_function(filename):
    """ Load tabulated transmission function from a text file and return
    linear intperpolator.

    Parameters
    ----------
    filename

    Returns
    --------
    interpolator
    """
    data = np.loadtxt(filename, unpack=True)
    assert len(data) > 1
    x = data[0]

    step = x[1:] - x[:-1]
    if not np.allclose(step[0], step):
        interpfunc = scipy_interpolate.interp1d(x, data[1])
        new_x = np.linspace(x.min(), x.max(), len(x))
        y = interpfunc(new_x)
        data = [0, y]
        x = new_x


    interp_funcs = []

    for i in range(1, len(data)):
        interp_funcs.append(interpolate.interpolate_regular(x, data[i], fill_low=0.0, fill_high=0.0))

    if len(interp_funcs) == 1:
        return interp_funcs[0]

    return interp_funcs


class Optics(object):
    """ Describes the optics of the telescope that has been read from a configuration file. """

    logger = logging.getLogger(__name__)

    params = ('collecting_surface_area', 'pix_size', 'pix_disp', 'lambda_range', 'transmission_path', 'psf_amp', 'psf_sig1', 'psf_sig2')

    def __init__(self, config=None, **kwargs):
        """ """
        config = config if config else {}

        self.config = config

        # merge key,value arguments and config dict
        for key, value in kwargs.items():
            self.config[key] = value

        self.collecting_area = self.config['collecting_surface_area']

        self.PSF = psf.PSF_model(config['psf_amp'], config['psf_sig1'], config['psf_sig2'])

        self.ARCSEC_TO_PIX = 1. / self.config['pix_size']
        self.PIX_TO_ARCSEC = self.config['pix_size']

        self.lambda_start, self.lambda_end = self.config['lambda_range']
        self.lambda_range = self.lambda_end - self.lambda_start
        self.lambda_ref = (self.lambda_start + self.lambda_end)/2.

        # recompute length of spectrum in degrees (todo: change to pixel coordinates)
        self.grism_transmission = {}
        self.grism_transmission[1] = load_transmission_function(self.config['transmission_path'])

    def transmission(self, wavelength, order=1):
        """ Compute transmission at wavelength

        Inputs
        ------
        wavelength (angstroms)

        Outputs
        -------
        transmission
        """
        return self.grism_transmission[order].evaluate(wavelength)

    def integrate(self, func, wavelength0=None, wavelength1=None, order=1):
        """ Integrate a function over the transmission curve

        Notes
        -----
        Assumes that func is in flux units and converts to photon counts.

        Parameters
        ----------
        func : object
            function to integrate should take 1 argument (wavelength in angstrom)
        wavelength0 : float
            start of integration range
        wavelength1 : float
            end of inegration range
        order : int
            dispersion order

        Returns
        -------
        counts : float
        """
        f = lambda x: phot.flux_to_photon(func(x), self.collecting_area, x) * self.grism_transmission[order].scalar(x)
        if wavelength0 is None:
            wavelength0 = self.lambda_start
        if wavelength1 is None:
            wavelength1 = self.lambda_end
        return integrate.quad(f, wavelength0, wavelength1, epsrel=1e-3)[0]

