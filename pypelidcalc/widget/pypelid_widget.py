import time

import numpy as np
from templatefit import template_fit
from IPython.display import display
from ipywidgets import HTML, HBox, Button, Tab, Output, IntProgress
from scipy import interpolate

import instrument_widget, foreground_widget, galaxy_widget, analysis_widget, survey_widget, config_widget
import pypelidcalc
from pypelidcalc.spectra import galaxy, linesim
from pypelidcalc.survey import phot, optics
from pypelidcalc.utils import consts

import plotly.graph_objects as go


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


def centroidz(z, pz, window=5):
    """ """
    ibest = np.argmax(pz)
    low = max(0, ibest - window)
    high = min(len(pz), ibest + window + 1)
    sel = slice(low, high)
    z = z[sel]
    pz = pz[sel]
    return np.sum(z * pz)/np.sum(pz)


class PypelidWidget(object):
    """ """
    def __init__(self):
        self.instrument = instrument_widget.Instrument()
        self.foreground = foreground_widget.Foreground()
        self.galaxy = galaxy_widget.Galaxy()
        self.analysis = analysis_widget.Analysis()
        self.survey = survey_widget.Survey()
        self.config = config_widget.Config((self.galaxy, self.foreground, self.instrument, self.survey, self.analysis))
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

            O = optics.Optics(config, seed=time.time()*1e6)

            L = linesim.LineSimulator(O, extraction_sigma=self.analysis.widgets['extraction_sigma'].value)

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
                signal = phot.flux_to_photon(flux, O.collecting_area, wavelength)
                signal *= exp_time * nexp
                signal *= O.transmission(np.array([wavelength]), 1)[0]

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


        zgrid = np.arange(self.analysis.widgets['zmin'].value, self.analysis.widgets['zmax'].value,self.analysis.widgets['zstep'].value)
        zfitter = template_fit.TemplateFit(wavelength_scale, zgrid, consts.line_list,
                    template_file=self.analysis.template_path)


        nloops = self.analysis.widgets['nloops'].value
        self.progress.min=0
        self.progress.max=nloops

        realizations = [[] for v in obs_list]
        real_stack = []

        prob_z = []
        zmeas = []

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
            pz = np.array(zfitter.pz())
            prob_z.append(pz)
            zmeas.append(centroidz(zgrid, pz))

            self.progress.value = loop


        m = np.mean(real_stack, axis=0)
        var = np.var(real_stack, axis=0)

        with self.plot:
            self.plot.clear_output()

            fig = go.Figure(data=go.Scatter(x=wavelength_scale/1e4, y=m, name='Signal'))
            fig.add_trace(go.Scatter(x=wavelength_scale/1e4, y=var**0.5, name='Noise'))
            fig.update_layout(xaxis_title='Wavelength (micron)',
                              yaxis_title='Flux density',margin=dict(l=0, r=0, t=0, b=80, pad=0))

            ii = var>0
            print "SNR stack: %g"%np.sqrt(np.sum(m[ii]**2/var[ii]))

            a = np.max(m)
            b = np.median(var)**0.5

            display(fig)

            h, e = np.histogram(zmeas, bins=zgrid)
            h = h * 1./ np.sum(h)
            x = (e[1:]+e[:-1])/2.

            fig2 = go.Figure(data=go.Scatter(x=x, y=h, name='Measured redshift'))
            fig2.add_trace(go.Scatter(x=zgrid, y=np.mean(prob_z, axis=0), name='p(z)'))
            fig2.update_layout(xaxis_title='Redshift',
                              yaxis_title='Distribution',margin=dict(l=0, r=0, t=0, b=80, pad=0))

            display(fig2)




        button.disabled = False

    def tab_event(self, change):
        if change['type'] == 'change' and change['name'] == 'selected_index':
            if change['new'] == 2:
                self.instrument.plot_transmission()
                self.instrument.plot_psf()
            elif change['new'] == 5:
                self.config.update()



    def show(self):
        """ """
        display(HTML("Pypelidcalc version: %s"%pypelidcalc.__version__))
        tab = Tab([self.galaxy.widget, self.foreground.widget, self.instrument.widget, self.survey.widget, self.analysis.widget, self.config.widget])
        tab.set_title(0, "Source")
        tab.set_title(1, "Foreground")
        tab.set_title(2, "Instrument")
        tab.set_title(3, "Survey")
        tab.set_title(4, "Analysis")
        tab.set_title(5, "Config")

        tab.observe(self.tab_event)

        display(tab)

        button = Button(description="Compute", icon='play')

        button.on_click(self.run)

        display(HBox([button, self.progress]))
        display(self.plot)