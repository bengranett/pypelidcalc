import os
import numpy as np
from IPython.display import display,clear_output
from ipywidgets import HTML, HBox, VBox, BoundedFloatText, Dropdown, Output, Label
import plotly.graph_objects as go

from pypelidcalc.survey import psf


configurations = {
    'Euclid NISP': {
            'collecting_surface_area': 10000,
            'pix_size': 0.3,
            'pix_disp': 13.4,
            'psf_amp': 0.781749,
            'psf_sig1': 0.84454,
            'psf_sig2': 3.6498,
            'readnoise': 8.87,
            'darkcurrent': 0.019,
            'transmission_red': 'red_transmission.txt',
            'transmission_blue': 'blue_transmission.txt',
    },
    'HST G141': {
            'collecting_surface_area': 45239,
            'psf_amp': 1,
            'psf_sig1': 0.4,
            'psf_sig2': 0,
            'pix_size': 0.13,
            'pix_disp': 24.,
            'readnoise': 12,
            'darkcurrent': 0.048,
            'transmission_red': 'HST_WFC3_IR.G141.dat',
            'transmission_blue': 'none',

    },
    'custom': {},
}

TRANSMISSION_DIR='in'

def get_transmission_files(dir=TRANSMISSION_DIR):
    filelist = []
    if not os.path.exists(dir):
        return filelist
    for filename in os.listdir(dir):
        if filename.endswith("txt") or filename.endswith("dat"):
            filelist.append(filename)
    filelist += ['none']
    return filelist


class Instrument(object):
    """ """
    style = {'description_width': '150px'}
    layout = {'width': '250px'}

    widgets = {
        'config': Dropdown(options=configurations.keys(), description='Configurations:'),
        'collecting_surface_area': BoundedFloatText(value=10000, min=0, max=1e6, step=10, description='Collecting area (cm2)'),
        'pix_size': BoundedFloatText(value=0.3, min=0, max=10, step=0.1, description='Pixel size (arcsec)'),
        'pix_disp': BoundedFloatText(value=13.4, min=0, max=100, step=0.1, description='Dispersion (A/pixel)'),
        'psf_amp': BoundedFloatText(value=0.781749, min=0, max=1, step=0.05, description='PSF amplitude'),
        'psf_sig1': BoundedFloatText(value=0.84454, min=0, max=20, step=0.1, description='PSF sigma 1 (pixels)'),
        'psf_sig2': BoundedFloatText(value=3.6498, min=0, max=20, step=0.1, description='PSF sigma 2 (pixels)'),
        'readnoise': BoundedFloatText(value=8.87, min=0, max=100, step=0.1, description='read noise (electrons)'),
        'darkcurrent': BoundedFloatText(value=0.019, min=0, max=100, step=0.1, description='dark current (elec/s/pix)'),
        'transmission_red': Dropdown(options=get_transmission_files(), description='Red grism transmission'),
        'transmission_blue': Dropdown(options=get_transmission_files(), description='Blue grism transmission'),
        'radius1': Label(),
        'radius2': Label(),

    }

    params = ('collecting_surface_area', 'pix_size', 'pix_disp',
              'psf_amp', 'psf_sig1', 'psf_sig2',
              'readnoise', 'darkcurrent',
              'transmission_red', 'transmission_blue')

    def __init__(self):
        self._set_custom = True

        for key, widget in self.widgets.items():
            widget.style = self.style
            widget.layout = self.layout

        self.update(**configurations[self.widgets['config'].value])

        self.widgets['config'].observe(self.change_config, names='value')
        self.widgets['transmission_red'].observe(self.plot_transmission, names='value')
        self.widgets['transmission_blue'].observe(self.plot_transmission, names='value')

        self.widgets['psf_amp'].observe(self.plot_psf, names='value')
        self.widgets['psf_sig1'].observe(self.plot_psf, names='value')
        self.widgets['psf_sig2'].observe(self.plot_psf, names='value')

        for key, widget in self.widgets.items():
            if key in ['config', 'transmission_red', 'transmission_blue']:
                continue
            widget.observe(self.modify, names='value')

        self.figs = [go.FigureWidget(), go.FigureWidget()]
        self.figs[0].update_layout(xaxis_title='Wavelength (microns)',
                          yaxis_title='Throughput',
                          width=500, height=200,
                          margin=dict(l=0, r=0, t=0, b=0, pad=0), autosize=True)
        self.figs[1].update_layout(xaxis_title='Radius (pixels)',
                          yaxis_title='PSF', width=500, height=200,
                          margin=dict(l=0, r=0, t=0, b=0, pad=0), autosize=True)


        title = HTML('<h2>Instrument<h2>')
        elements = []
        elements.append(self.widgets['config'])
        elements += [HTML('<b>Optics</b>'), self.widgets['collecting_surface_area'], self.widgets['pix_size'], self.widgets['pix_disp'],
        self.widgets['transmission_red'],self.widgets['transmission_blue']]
        elements += [HTML('<b>PSF</b>'), self.widgets['psf_amp'], self.widgets['psf_sig1'], self.widgets['psf_sig2']]
        elements += [HTML('<b>Detector</b>'), self.widgets['readnoise'], self.widgets['darkcurrent']]
        self.widget = HBox(
            [VBox(elements),
            VBox(self.figs + [self.widgets['radius1'], self.widgets['radius2']])]
        )

        self.plot_transmission()
        self.plot_psf()


    def update(self, **kwargs):
        """ """
        for key, value in kwargs.items():
            if key in self.widgets:
                self.widgets[key].value = value

    def change_config(self, change):
        """ """
        self._set_custom = False
        key = change['new']
        self.update(**configurations[key])
        self._set_custom = True

    def modify(self, change):
        """ """
        if not self._set_custom:
            return
        self.widgets['config'].value = 'custom'


    def plot_transmission(self, change=None):
        """ """
        if change:
            self.widgets['config'].value = 'custom'

        colors = {'transmission_red':'red', 'transmission_blue': 'blue'}

        fig = self.figs[0]
        fig.data = []

        for key in ['transmission_red', 'transmission_blue']:

            if not self.widgets[key].value:
                continue

            if self.widgets[key].value == 'none':
                continue

            path = os.path.join(TRANSMISSION_DIR, self.widgets[key].value)
            if not os.path.exists(path):
                continue
            x, y = np.loadtxt(path, unpack=True)

            sel, = np.where(y > (y.max()/10.))
            a = max(0, sel[0] - 10)
            b = min(len(x), sel[-1] + 10)
            x = x[a:b]
            y = y[a:b]

            fig.add_scatter(x=x/1e4, y=y, name=self.widgets[key].value, line_color=colors[key])


    def plot_psf(self, change=None):
        """ """
        if change:
            self.widgets['config'].value = 'custom'

        sig1 = self.widgets['psf_sig1'].value
        sig2 = self.widgets['psf_sig2'].value
        amp = self.widgets['psf_amp'].value

        PSF = psf.PSF_model(amp, sig1, sig2)

        r1 = PSF.radius(0.5)
        r2 = PSF.radius(0.8)
        self.widgets['radius1'].value = "PSF 50%%-radius: %3.2f pixels"%r1
        self.widgets['radius2'].value = "PSF 80%%-radius: %3.2f pixels"%r2


        x = np.linspace(0,np.ceil(PSF.radius(0.95)),100)
        y = np.array(PSF.prof(x))
        yinteg = np.array(PSF.evaluate(x))

        fig = self.figs[1]
        fig.data = []
        fig.add_scatter(x=x, y=y/y.max(), name='Profile')
        fig.add_scatter(x=x, y=yinteg, name='Integrated')


    def get_lambda_range(self, key, thresh=2):
        """"""
        path = os.path.join(TRANSMISSION_DIR, self.widgets[key].value)
        x, y = np.loadtxt(path , unpack=True)
        sel = y > (y.max()*1./thresh)
        x = x[sel]
        return x.min(), x.max()

    def get_config_list(self):
        """ """
        config_list = []
        for key in 'transmission_red', 'transmission_blue':
            if not self.widgets[key].value or self.widgets[key].value == 'none':
                continue
            config = {}
            config['collecting_surface_area'] = self.widgets['collecting_surface_area'].value
            config['pix_size'] = self.widgets['pix_size'].value
            config['pix_disp'] = [1, self.widgets['pix_disp'].value, 1]
            config['psf_amp'] = self.widgets['psf_amp'].value
            config['psf_sig1'] = self.widgets['psf_sig1'].value
            config['psf_sig2'] = self.widgets['psf_sig2'].value
            config['readnoise'] = self.widgets['readnoise'].value
            config['darkcurrent'] = self.widgets['darkcurrent'].value
            config['transmission_path'] = os.path.join(TRANSMISSION_DIR, self.widgets[key].value)
            config['lambda_range'] = self.get_lambda_range(key)
            config_list.append(config)
        return config_list
