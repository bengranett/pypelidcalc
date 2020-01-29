import time
import threading

import numpy as np
from templatefit import template_fit
from IPython.display import display
from ipywidgets import HTML, HBox, VBox, Button, Tab, Output, IntProgress, Label, BoundedIntText, IntText, Checkbox
from scipy import interpolate

from . import instrument_widget, foreground_widget, galaxy_widget, analysis_widget, survey_widget, config_widget
import pypelidcalc
from pypelidcalc.spectra import galaxy, linesim
from pypelidcalc.survey import phot, optics
from pypelidcalc.utils import consts
from pypelidcalc.cutils import rng

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
        'signal_on': Checkbox(value=True, description='Signal', layout={'width':'80px'}, style={'description_width': '0px'}),
        'noise_on': Checkbox(value=True, description='Noise', layout={'width':'80px'}, style={'description_width': '0px'}),
        'real_on': Checkbox(value=True, description='Realization', layout={'width':'80px'}, style={'description_width': '0px'}),
        'seed': IntText(description='Seed', disabled=True, layout={'width':'150px'}, style={'description_width': '50px'}),
        'seed_checkbox': Checkbox(value=False, description='Freeze random seed',layout={'width':'150px'}, style={'description_width': '0px'}),
    }

    def __init__(self):
        self.instrument = instrument_widget.Instrument()
        self.foreground = foreground_widget.Foreground()
        self.galaxy = galaxy_widget.Galaxy()
        self.analysis = analysis_widget.Analysis()
        self.survey = survey_widget.Survey()
        self.config = config_widget.Config((self.galaxy, self.foreground, self.instrument, self.survey, self.analysis))
        self.running = False

        self.render_lock = threading.Lock()
        self.param_lock = threading.Lock()

    def render(self, change=None):
        """ """
        if not self.render_lock.acquire(False):
            # print "locked"
            return
        if not self.param_lock.acquire(False):
            return
        render_thread = threading.Thread(target=self._render, args=((self.render_lock, self.param_lock),))
        render_thread.start()

    def _render(self, locks):
        """ """
        self.widgets['render_button'].style.button_color = 'orange'

        if not self.widgets['seed_checkbox'].value:
            self.widgets['seed'].value = np.random.randint(0,1e6)
        seed = self.widgets['seed'].value

        rng.seed(seed)

        wavelength_scale, flux, var, obs_list = self.spec(noise=False)
        wavelength_scale_, flux_n, var_, obs_list_ = self.spec(noise=True)

        self.wavelength_scale = wavelength_scale / 1e4
        step = wavelength_scale[1] - wavelength_scale[0]
        self.signal = flux / step
        self.real = flux_n / step
        self.noise = var**0.5 / step

        self.hideshow_line()

        L, gal = obs_list[0]
        x, y = np.transpose(gal.sample(int(1e6), L.plate_scale, self.galaxy.widgets['iso'].value))

        dx, dy = np.transpose(L.PSF.sample(len(x)))

        x += dx
        y += dy

        r = np.sqrt(x*x + y*y)
        w = int(np.ceil(np.percentile(r, 80))) + 0.5
        w = min(20.5, w)
        b = np.arange(-w, w+1, 1)
        h, ey, ex = np.histogram2d(y, x, bins=(b, b))

        bc = (ey[1:]+ey[:-1])/2.

        self.figs['image'].data[0]['z'] = h
        self.figs['image'].data[0]['x'] = bc
        self.figs['image'].data[0]['y'] = bc

        ii = var > 0
        snr = np.sqrt(np.sum(flux[ii]**2/var[ii]))
        self.widgets['snrbox'].value = "%3.2f"%snr

        self.widgets['render_button'].style.button_color = 'lightgreen'
        for lock in locks:
            lock.release()



    def spec(self, noise=True):
        emission_lines = [
            ('Ha', self.galaxy.widgets['flux_ha'].value * 1e-16),
            ('N2a', self.galaxy.widgets['flux_n2a'].value * 1e-16),
            ('N2b', self.galaxy.widgets['flux_n2b'].value * 1e-16),
            ('S2a', self.galaxy.widgets['flux_s2a'].value * 1e-16),
            ('S2b', self.galaxy.widgets['flux_s2b'].value * 1e-16),
            ('S3a', self.galaxy.widgets['flux_s3a'].value * 1e-16),
            ('S3b', self.galaxy.widgets['flux_s3b'].value * 1e-16),
            ('O3a', self.galaxy.widgets['flux_o3a'].value * 1e-16),
            ('O3b', self.galaxy.widgets['flux_o3b'].value * 1e-16),
            ('Hb', self.galaxy.widgets['flux_hb'].value * 1e-16),
            ('O2', self.galaxy.widgets['flux_o2'].value * 1e-16),

        ]


        nexp_list = self.survey.widgets['nexp_red'].value, self.survey.widgets['nexp_blue'].value
        exp_time = self.survey.widgets['exp_time'].value

        ztol = self.analysis.widgets['ztol'].value



        config_list = self.instrument.get_config_list()
        obs_list = []
        for i, config in enumerate(config_list):

            nexp = nexp_list[i]
            if nexp == 0:
                continue

            O = optics.Optics(config)

            L = linesim.LineSimulator(O, extraction_sigma=self.analysis.widgets['extraction_sigma'].value, isotropize=self.galaxy.widgets['iso'].value)

            det_bg = nexp * exp_time * config['darkcurrent'] + config['readnoise']**2

            det_bg += nexp * exp_time * self.foreground.widgets['foreground'].value

            gal = galaxy.Galaxy(
                z=self.galaxy.widgets['redshift'].value,
                bulge_scale=self.galaxy.widgets['bulge_scale'].value,
                disk_scale=self.galaxy.widgets['disk_scale'].value,
                bulge_fraction=self.galaxy.widgets['bulge_fraction'].value,
                axis_ratio=self.galaxy.widgets['axis_ratio'].value,
                pa=self.galaxy.widgets['pa'].value,
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

                if noise is False:
                    v = (signal * scale)**2/1e7
                else:
                    v = signal * scale**2

                gal.append_line(
                    wavelength=consts.line_list[line],
                    flux=signal * scale,
                    variance=v,
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

        specset = []
        for i, obs in enumerate(obs_list):
            L, gal = obs
            spectra = L.sample_spectrum(gal)
            if noise:
                s = spectra[0]
            else:
                s = spectra[1]
            specset.append((wavelength_scales[i], np.array(s), np.array(spectra[2])))
        flux_stack, var_stack = combine_spectra(wavelength_scale, specset)

        return wavelength_scale, flux_stack, var_stack, obs_list


    def update(self, zgrid, zmeas, wavelength_scale, mean_total, var_total, count):
        """ """
        zmeas = np.array(zmeas)

        m = mean_total * 1./ count
        var = var_total * 1./ count - m**2

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

        self.figs['pdf'].data[0]['x'] = x
        self.figs['pdf'].data[0]['y'] = h


    def run(self, stop_event):
        """ """
        self.param_lock.acquire()
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

            O = optics.Optics(config)

            L = linesim.LineSimulator(O, extraction_sigma=self.analysis.widgets['extraction_sigma'].value, isotropize=self.galaxy.widgets['iso'].value)

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


        prob_z = []
        zmeas = []

        t0 = time.time()
        t1 = time.time()

        mean_total = 0
        var_total = 0
        count = 0

        for loop in range(nloops):
            if stop_event.is_set():
                break
            specset = []
            for i, obs in enumerate(obs_list):
                L, gal = obs
                spectra = L.sample_spectrum(gal)
                specset.append((wavelength_scales[i], np.array(spectra[0]), np.array(spectra[2])))
            flux_stack, var_stack = combine_spectra(wavelength_scale, specset)

            mean_total += flux_stack
            var_total += flux_stack**2
            count += 1

            ii = var_stack>0
            invvar = np.zeros(len(var_stack), dtype='d')
            invvar[ii] = 1./var_stack[ii]

            amp = zfitter.template_fit(flux_stack, invvar, 2)
            pz = np.array(zfitter.pz())
            zmeas.append(centroidz(zgrid, pz))

            if time.time()-t0 > 10:
                self.update(zgrid, zmeas, wavelength_scale, mean_total, var_total, count)
                t0 = time.time()

            if time.time()-t1 > 1:
                self.widgets['progress'].value = loop
                self.widgets['progress'].description = "%i/%i"%(loop+1, nloops)
                self.widgets['timer'].value = "elapsed time: %i s"%(time.time()-self._start_time)
                t1 = time.time()

        self.widgets['progress'].description = "%i/%i"%(loop+1, nloops)
        self.widgets['timer'].value = "elapsed time: %i s"%(time.time()-self._start_time)

        self.widgets['progress'].value = 0
        self.update(zgrid, zmeas, wavelength_scale, mean_total, var_total, count)
        self.reset_button(self.widgets['button'])
        self.param_lock.release()


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

    def hideshow_line(self, change=None):
        for key,i,arr in [('signal_on',2,self.signal),('real_on',1,self.real),('noise_on',0,self.noise)]:
            if self.widgets[key].value:
                if len(self.figs['spec'].data[i]['x']) != len(self.wavelength_scale):
                    self.figs['spec'].data[i]['x'] = self.wavelength_scale
                self.figs['spec'].data[i]['y'] = arr
            else:
                self.figs['spec'].data[i]['y'] = []

    def seed_checkbox(self, change=None):
        if self.widgets['seed_checkbox'].value:
            self.widgets['seed'].disabled = False
        else:
            self.widgets['seed'].disabled = True

    def show(self):
        """ """
        # display()
        about = VBox([HTML("<a href=\"https://github.com/bengranett/pypelidcalc\" target=\"_blank\">Pypelid-calc</a> version: %s"%pypelidcalc.__version__)])

        tab = Tab([self.galaxy.widget, self.foreground.widget, self.instrument.widget, self.survey.widget, self.analysis.widget, self.config.widget, about])
        tab.set_title(0, "Source")
        tab.set_title(1, "Foreground")
        tab.set_title(2, "Instrument")
        tab.set_title(3, "Survey")
        tab.set_title(4, "Analysis")
        tab.set_title(5, "Config")
        tab.set_title(6, "About")
        tab.layout={'height': '300px'}

        tab.observe(self.tab_event)

        display(tab)

        for group in [self.galaxy, self.foreground, self.instrument, self.survey, self.analysis]:
            for key, w in group.widgets.items():
                w.observe(self.render, names='value')

        self.figs = {}
        self.figs['spec'] = go.FigureWidget()
        self.figs['spec'].update_layout(xaxis_title=u'Wavelength (\u03BCm)',
                                  height=200,
                                  yaxis_title='Flux density',
                                  margin=dict(l=0, r=0, t=0, b=0, pad=0))
        self.figs['spec'].add_scatter(x=[], y=[], name='Noise', line_color='grey')
        self.figs['spec'].add_scatter(x=[], y=[], name='Realization', line_color='dodgerblue')
        self.figs['spec'].add_scatter(x=[], y=[], name='Signal', line_color='black')


        self.figs['image'] = go.FigureWidget()
        self.figs['image'].update_layout(height=200, width=200, margin=dict(l=0, r=0, t=0, b=0, pad=0))

        self.figs['image'].add_trace(go.Heatmap(z=[[]], showscale=False))

        self.widgets['render_button'] = Button(description="Update realization", layout={'border':'solid 1px black', 'width': '200px'})
        self.widgets['render_button'].on_click(self.render)

        self.widgets['seed_checkbox'].observe(self.seed_checkbox, names='value')

        self.widgets['signal_on'].observe(self.hideshow_line, names='value')
        self.widgets['real_on'].observe(self.hideshow_line, names='value')
        self.widgets['noise_on'].observe(self.hideshow_line, names='value')

        checkboxes = HBox([self.widgets['signal_on'], self.widgets['noise_on'], self.widgets['real_on']])
        display(HTML('<h3>Spectrum</h3>'))
        display(HBox([self.widgets['seed_checkbox'], self.widgets['seed']]))
        display(HBox([HTML('SNR:'), self.widgets['snrbox'], self.widgets['render_button'], checkboxes]))
        display(HBox([self.figs['spec'], self.figs['image']]))

        self.reset_button(self.widgets['button'])
        self.widgets['button'].on_click(self.click_start)

        elements = [HTML("<h3>Redshift measurement</h3>")]
        elements +=  [HBox([self.widgets['nreal'], self.widgets['button'], self.widgets['progress'], self.widgets['timer']])]

        horiz = [HTML('<b>Statistics:</b>')]
        horiz += [HTML('Mean z:'), self.widgets['zmeas']]
        horiz += [HTML('Error:'), self.widgets['zerr']]
        horiz += [HTML('68% limit:'), self.widgets['zerr_68']]
        horiz += [HTML('Fractional systematic:'), self.widgets['zerr_sys']]
        horiz += [HTML('Outlier rate:'), self.widgets['zerr_cat']]

        elements += [HBox(horiz)]

        display(VBox(elements))

        self.figs['pdf'] = go.FigureWidget()
        self.figs['pdf'].update_layout(xaxis_title='Redshift', height=200,
                              yaxis_title='Distribution',margin=dict(l=0, r=0, t=0, b=0, pad=0))

        self.figs['pdf'].add_scatter(x=[], y=[], name='Measured redshift')

        display(self.figs['pdf'])

        self.render_lock.acquire()
        self.param_lock.acquire()
        self._render((self.render_lock, self.param_lock))

