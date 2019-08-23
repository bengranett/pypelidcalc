import time
import threading

import numpy as np
from templatefit import template_fit
from IPython.display import display
from ipywidgets import HTML, HBox, VBox, Button, Tab, Output, IntProgress, Label, BoundedIntText
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

    widgets = {
        'nreal': BoundedIntText(value=1000, min=0, max=100000, step=100, description='Number of realizations:',
            layout={'width': '250px'},
            style={'description_width': '150px',}),
        'button': Button(description="Run", icon='play', layout={'border':'solid 1px black', 'width': '100px'}),
        'progress': IntProgress(bar_style='success'),
        'timer': Label(),
        'snrbox': Label(layout={'border':'solid 1px green', 'width': '100px'}),
        'zmeas': Label(layout={'border':'solid 1px green', 'width': '100px'}),
        'zerr': Label(layout={'border':'solid 1px green', 'width': '100px'}),
        'zerr_68': Label(layout={'border':'solid 1px green', 'width': '100px'}),
        'zerr_sys': Label(layout={'border':'solid 1px green', 'width': '100px'}),
        'zerr_cat': Label(layout={'border':'solid 1px green', 'width': '100px'}),

    }

    def __init__(self):
        self.instrument = instrument_widget.Instrument()
        self.foreground = foreground_widget.Foreground()
        self.galaxy = galaxy_widget.Galaxy()
        self.analysis = analysis_widget.Analysis()
        self.survey = survey_widget.Survey()
        self.config = config_widget.Config((self.galaxy, self.foreground, self.instrument, self.survey, self.analysis))
        self.plot = Output(layout={'width':'1000px'})
        self.running = False

    def update(self, zgrid, zmeas, wavelength_scale, real_stack):
        """ """
        zmeas = np.array(zmeas)
        m = np.mean(real_stack, axis=0)
        var = np.var(real_stack, axis=0)


        ii = var > 0
        snr = np.sqrt(np.sum(m[ii]**2/var[ii]))
        self.widgets['snrbox'].value = "%3.2f"%snr

        ztrue = self.galaxy.widgets['redshift'].value

        ztol = self.analysis.widgets['ztol'].value

        dz = np.abs(zmeas - ztrue)
        sel = dz < ztol
        if np.sum(sel) > 0:
            z = np.mean(zmeas[sel])
            dzobs = np.abs(zmeas - z)
            dz68 = np.percentile(dzobs[sel], 68)
            self.widgets['zerr_68'].value = "%3.2e"%dz68
            if dz68 > 0:
                self.widgets['zerr_sys'].value = "%g"%((ztrue-z)*np.sqrt(np.sum(sel))/dz68)
            self.widgets['zerr_cat'].value = "%f"%(1 - np.sum(sel) * 1. / len(zmeas))

            self.widgets['zmeas'].value = "%g"%z
            self.widgets['zerr'].value = "%3.2e"%(ztrue - z)

        h, e = np.histogram(zmeas, bins=zgrid)
        h = h * 1./ np.sum(h)
        x = (e[1:]+e[:-1])/2.
        a = np.where(h>0)[0][0]-1
        b = np.where(h>0)[0][-1]+1
        x = x[a:b+1]
        h = h[a:b+1]

        self.figs['spec'].data[0]['x'] = wavelength_scale/1e4
        self.figs['spec'].data[0]['y'] = m
        self.figs['spec'].data[1]['x'] = wavelength_scale/1e4
        self.figs['spec'].data[1]['y'] = var**0.5

        self.figs['pdf'].data[0]['x'] = x
        self.figs['pdf'].data[0]['y'] = h


    def run(self, stop_event):
        """ """
        self._start_time = time.time()

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

        ztol = self.analysis.widgets['ztol'].value

        self.figs['pdf'].update_layout(
            shapes=[go.layout.Shape(
                   type="rect",
                   xref="x",
                    yref="paper",
                    x0=self.galaxy.widgets['redshift'].value-ztol,
                    y0=0,
                    x1=self.galaxy.widgets['redshift'].value+ztol,
                    y1=1,
                    fillcolor="LightSalmon",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
        ),])



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

                if wavelength < O.lambda_start or wavelength > O.lambda_end:
                    continue

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
                    background=det_bg * scale**2,
                    rest_frame=1
                )

            # add a line at the center of the bandpass (observed frame)
            scale = phot.flux_to_photon(1, O.collecting_area, O.lambda_ref)
            scale *= exp_time * nexp
            scale *= O.transmission(np.array([O.lambda_ref]), 1)[0]
            scale = 1./scale
            gal.append_line(
                wavelength=O.lambda_ref,
                flux=0,
                variance=0,
                background=det_bg * scale**2,
                rest_frame=0
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
        zfitter = template_fit.TemplateFit(wavelength_scale, zgrid, consts.line_list, res=self.analysis.widgets['templ_res'].value,
                    template_file=self.analysis.template_path)


        nloops = self.widgets['nreal'].value
        self.widgets['progress'].min=0
        self.widgets['progress'].max=nloops

        realizations = [[] for v in obs_list]
        real_stack = []

        prob_z = []
        zmeas = []

        t0 = time.time()

        for loop in range(nloops):
            if stop_event.is_set():
                break
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
            # prob_z.append(pz)
            zmeas.append(centroidz(zgrid, pz))

            if time.time()-t0 > 10:
                self.update(zgrid, zmeas, wavelength_scale, real_stack)
                t0 = time.time()

            self.widgets['progress'].value = loop
            self.widgets['progress'].description = "%i/%i"%(loop+1, nloops)
            self.widgets['timer'].value = "elapsed time: %i s"%(time.time()-self._start_time)


        self.widgets['progress'].value = 0
        self.update(zgrid, zmeas, wavelength_scale, real_stack)
        self.reset_button(self.widgets['button'])


    def click_start(self, button):
        if not self.running:
            self.running = True
            button.description = "Stop"
            button.icon = "stop"
            button.style.button_color = 'orange'
            self.stop_event = threading.Event()
            thread = threading.Thread(target=self.run, args=(self.stop_event,))
            thread.start()
        else:
            self.stop_event.set()
            self.reset_button(button)

    def reset_button(self, button):
        """ """
        self.running = False
        button.description = "Run"
        button.icon = "play"
        button.style.button_color = 'lightgreen'

    def tab_event(self, change):
        if change['type'] == 'change' and change['name'] == 'selected_index':
            if change['new'] == 5:
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

        self.reset_button(self.widgets['button'])

        self.widgets['button'].on_click(self.click_start)

        elements =  [HBox([self.widgets['nreal'], self.widgets['button'], self.widgets['progress'], self.widgets['timer']])]
        elements += [HTML('<b>Redshift measurement statistics</b>')]
        elements += [HBox([HTML('<b>SNR:</b>'), self.widgets['snrbox']])]
        horiz = [HTML('<b>Mean z:</b>'), self.widgets['zmeas']]
        horiz += [HTML('<b>Error:</b>'), self.widgets['zerr']]
        horiz += [HTML('<b>68% limit:</b>'), self.widgets['zerr_68']]
        horiz += [HTML('<b>Fractional systematic:</b>'), self.widgets['zerr_sys']]
        horiz += [HTML('<b>Outlier rate:</b>'), self.widgets['zerr_cat']]

        elements += [HBox(horiz)]

        display(VBox(elements))

        display(self.plot)

        self.figs = {}
        with self.plot:
            self.figs['spec'] = go.FigureWidget()
            self.figs['pdf'] = go.FigureWidget()
            self.figs['spec'].update_layout(xaxis_title='Wavelength (micron)',
                              height=200,
                              yaxis_title='Flux density',margin=dict(l=0, r=0, t=0, b=0, pad=0))

            self.figs['pdf'].update_layout(xaxis_title='Redshift', height=200,
                              yaxis_title='Distribution',margin=dict(l=0, r=0, t=0, b=0, pad=0))

            self.figs['spec'].add_scatter(x=[], y=[], name='Signal')
            self.figs['spec'].add_scatter(x=[], y=[], name='Noise')
            self.figs['pdf'].add_scatter(x=[], y=[], name='Measured redshift')

            display(self.figs['spec'])
            display(self.figs['pdf'])

