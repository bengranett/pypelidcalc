import time

import numpy as np
import templatefit
from IPython.display import display
from ipywidgets import HTML, HBox, Button, Tab, Output, IntProgress
from matplotlib import pyplot as plt
from scipy import interpolate

import instrument_widget, foreground_widget, galaxy_widget, analysis_widget, survey_widget
import pypelidcalc
from pypelidcalc.spectra import galaxy, linesim
from pypelidcalc.survey import phot
from pypelidcalc.utils import consts


plt.ioff()

def combine_spectra(wavelength_scale, specset):
    """ """
    fstack = np.zeros(len(wavelength_scale), dtype='d')
    norm = np.zeros(len(wavelength_scale), dtype='d')
    for scale, flux, variance in specset:
        ii = variance > 0
        f_interp = interpolate.interp1d(scale[ii], flux[ii]/variance[ii], bounds_error=False, fill_value=0)
        v_interp = interpolate.interp1d(scale[ii], 1./variance[ii], bounds_error=False, fill_value=0)
        fstack += f_interp(wavelength_scale)
        norm += v_interp(wavelength_scale)
    ii = norm > 0
    fstack[ii] /= norm[ii]
    norm[ii] = 1./norm[ii]
    return fstack, norm


class PypelidWidget(object):
    """ """
    def __init__(self):
        self.instrument = instrument_widget.Instrument()
        self.foreground = foreground_widget.Foreground()
        self.galaxy = galaxy_widget.Galaxy()
        self.analysis = analysis_widget.Analysis()
        self.survey = survey_widget.Survey()
        self.progress = IntProgress(bar_style='success')
        self.plot = Output(layout={'width':'1000px'})

    def run(self, button):
        """ """
        button.disabled = True

        emission_lines = [
            ('Ha', self.galaxy.widgets['flux_ha'].value * 1e-16),
            ('N2a', self.galaxy.widgets['flux_n2a'].value * 1e-16),
            ('N2b', self.galaxy.widgets['flux_n2b'].value * 1e-16),
            ('S2a', self.galaxy.widgets['flux_s2a'].value * 1e-16),
            ('S2b', self.galaxy.widgets['flux_s2b'].value * 1e-16),
            ('O3a', self.galaxy.widgets['flux_o3a'].value * 1e-16),
            ('O3b', self.galaxy.widgets['flux_o3b'].value * 1e-16),
            ('Hb', self.galaxy.widgets['flux_hb'].value * 1e-16),
            ('O2', self.galaxy.widgets['flux_o2'].value * 1e-16),

        ]


        nexp_list = self.survey.widgets['nexp_red'].value, self.survey.widgets['nexp_blue'].value
        exp_time = self.survey.widgets['exp_time'].value




        config_list = self.instrument.get_config_list()
        obs_list = []
        for i, config in enumerate(config_list):

            nexp = nexp_list[i]
            if nexp == 0:
                continue

            optics = optics.Optics(config, seed=time.time()*1e6)

            L = linesim.LineSimulator(optics, extraction_sigma=self.analysis.widgets['extraction_sigma'].value)

            det_bg = nexp * exp_time * config['darkcurrent'] + config['readnoise']**2

            det_bg += nexp * exp_time * self.foreground.widgets['foreground'].value

            gal = galaxy.Galaxy(
                z=self.galaxy.widgets['redshift'].value,
                bulge_scale=self.galaxy.widgets['bulge_scale'].value,
                disk_scale=self.galaxy.widgets['disk_scale'].value,
                bulge_fraction=self.galaxy.widgets['bulge_fraction'].value,
                axis_ratio=self.galaxy.widgets['axis_ratio'].value,
                velocity_disp=self.galaxy.widgets['velocity_dispersion'].value,
            )

            for line, flux in emission_lines:
                wavelength = (1 + gal.z) * consts.line_list[line]
                signal = phot.flux_to_photon(flux, optics.collecting_area, wavelength)
                signal *= exp_time * nexp
                signal *= optics.transmission(np.array([wavelength]), 1)[0]

                if signal <= 0:
                    continue

                line_variance = signal

                scale = flux / signal

                gal.append_line(
                    wavelength=consts.line_list[line],
                    flux=signal * scale,
                    variance=signal * scale**2,
                    background=det_bg * scale**2
                )
                gal.compute_obs_wavelengths(gal.z)

            if gal.line_count == 0:
                continue

            obs_list.append((L, gal))

        wavelength_scales = []
        dispersion = []
        for L, gal in obs_list:
            x = np.arange(L.npix) * L.dispersion + L.lambda_min
            wavelength_scales.append(x)
            dispersion.append(L.dispersion)
        dispersion = np.min(dispersion)
        wavelength_min = np.min(np.concatenate(wavelength_scales))
        wavelength_max = np.max(np.concatenate(wavelength_scales))
        wavelength_scale = np.arange(wavelength_min, wavelength_max, dispersion)


        zgrid = np.arange(0, 2,.001)
        zfitter = templatefit.template_fit.TemplateFit(wavelength_scale, zgrid,
                    template_file=self.analysis.template_path)


        nloops = self.analysis.widgets['nloops'].value
        self.progress.min=0
        self.progress.max=nloops

        realizations = [[] for v in obs_list]
        real_stack = []

        prob_z = []

        for loop in range(nloops):
            specset = []
            for i, obs in enumerate(obs_list):
                L, gal = obs
                spectra = L.sample_spectrum(gal)
                realizations[i].append(np.array(spectra[0]))
                specset.append((wavelength_scales[i], np.array(spectra[0]), np.array(spectra[2])))
            flux_stack, var_stack = combine_spectra(wavelength_scale, specset)
            real_stack.append(flux_stack)

            ii = var_stack>0
            invvar = np.zeros(len(var_stack), dtype='d')
            invvar[ii] = 1./var_stack[ii]

            amp = zfitter.template_fit(flux_stack, invvar, 2)
            prob_z.append(np.array(zfitter.pz()))

            self.progress.value = loop


        m = np.mean(real_stack, axis=0)
        var = np.var(real_stack, axis=0)

        with self.plot:
            self.plot.clear_output()
            fig = plt.figure(figsize=(16.5, 3))
            # limits = []
            # colors = ['hotpink', 'dodgerblue']
            # for i in range(len(m)):
            #     sig = var[i]**.5
            #     plt.fill_between(wavelength_scales[i], -sig, sig, color=colors[i], alpha=0.5)
            #     # plt.plot(wavelength_scales[i], m[i], lw=2, c=colors[i], zorder=10)
            #     # limits.append((m[i].max(), np.median(var[i])**.5))
            #     print "SNR %i: %g"%(i, np.sqrt(np.sum(m[i]**2/var[i])))

            plt.plot(wavelength_scale, m, lw=2, c='k', zorder=11)
            sig = var**0.5
            plt.fill_between(wavelength_scale, -sig, sig, color='grey', alpha=0.5)

            plt.grid(True)

            ii = var>0
            print "SNR stack: %g"%np.sqrt(np.sum(m[ii]**2/var[ii]))

            a = np.max(m)
            b = np.median(var)**0.5

            plt.ylim(-b, a+b)
            plt.xlim(wavelength_scale.min(), wavelength_scale.max())
            plt.xlabel("Wavelength (A)")
            plt.ylabel("Flux density")
            display(fig)
            # show_inline_matplotlib_plots()

            fig2 = plt.figure(figsize=(16.5, 3))
            plt.semilogy(zgrid, np.mean(prob_z, axis=0))
            plt.grid()
            plt.xlabel("Redshift")
            plt.ylabel("p(z)")
            display(fig2)


        button.disabled = False

    def show(self):
        """ """
        display(HTML("Pypelidcalc version: %s"%pypelidcalc.__version__))
        tab = Tab([self.galaxy.widget, self.foreground.widget, self.instrument.widget, self.survey.widget, self.analysis.widget])
        tab.set_title(0, "Source")
        tab.set_title(1, "Foreground")
        tab.set_title(2, "Instrument")
        tab.set_title(3, "Survey")
        tab.set_title(4, "Analysis")

        display(tab)

        button = Button(description="Compute", icon='play')

        button.on_click(self.run)

        display(HBox([button, self.progress]))
        display(self.plot)