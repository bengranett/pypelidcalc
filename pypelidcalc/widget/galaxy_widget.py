import numpy as np
from ipywidgets import Label, HTML, HBox, VBox, BoundedFloatText, Checkbox
from pypelidcalc.spectra.bulgy_disk import bulgy_disk_radius


class Galaxy(object):
    """ """

    style = {'description_width': '200px'}
    layout = {'width': '300px'}

    widgets = {
        'redshift': BoundedFloatText(value=1, min=0, max=10, step=0.1, description="Redshift"),
        'bulge_scale': BoundedFloatText(value=0.5, min=0, max=10, step=0.1, description="Bulge scale (arcsec)"),
        'disk_scale': BoundedFloatText(value=0.3, min=0, max=10, step=0.1, description="Disk scale (arcsec)"),
        'bulge_fraction': BoundedFloatText(value=.5, min=0, max=1, step=0.1, description="Bulge fraction"),
        'axis_ratio': BoundedFloatText(value=1, min=0.01, max=1, step=0.1, description="Axis ratio"),
        'pa': BoundedFloatText(value=0, min=-180, max=180, step=10, description="Position angle"),
        'iso': Checkbox(value=True, description="Isotropize"),
        'half_light_radius': HBox([Label("Half-light radius (arcsec): "), Label(value="0")]),
        'velocity_dispersion': BoundedFloatText(value=0, min=0, max=1000, step=0.1, description="Velocity dispersion (km/s)"),
        'flux_ha': BoundedFloatText(value=2, min=0, max=1000, step=0.1, description='Flux H$\\alpha$ 6565 ($10^{-16}$ erg/cm2/s):'),
        'n2_ha_ratio': BoundedFloatText(value=0, min=0, max=10, step=0.1, description='NII/H$\\alpha$:'),
        'flux_n2a': BoundedFloatText(value=0, min=0, max=1000, step=0.1, description='Flux NIIa 6550 ($10^{-16}$ erg/cm2/s):'),
        'flux_n2b': BoundedFloatText(value=0, min=0, max=1000, step=0.1, description='Flux NIIb 6585 ($10^{-16}$ erg/cm2/s):'),
        'flux_hb': BoundedFloatText(value=0, min=0, max=1000, step=0.1, description='Flux Hb 4863 ($10^{-16}$ erg/cm2/s):'),
        'flux_o3a': BoundedFloatText(value=0, min=0, max=1000, step=0.1, description='Flux OIIIa 4960 ($10^{-16}$ erg/cm2/s):'),
        'flux_o3b': BoundedFloatText(value=0, min=0, max=1000, step=0.1, description='Flux OIIIb 5008 ($10^{-16}$ erg/cm2/s):'),
        'flux_s2a': BoundedFloatText(value=0, min=0, max=1000, step=0.1, description='Flux SIIa 6718 ($10^{-16}$ erg/cm2/s):'),
        'flux_s2b': BoundedFloatText(value=0, min=0, max=1000, step=0.1, description='Flux SIIb 6733 ($10^{-16}$ erg/cm2/s):'),
        'flux_o2': BoundedFloatText(value=0, min=0, max=1000, step=0.1, description='Flux OII 3727 ($10^{-16}$ erg/cm2/s):'),
    }

    params = ('redshift', 'bulge_scale', 'disk_scale', 'bulge_fraction',
              'axis_ratio','pa','iso',
              'velocity_dispersion', 'flux_ha', 'flux_n2a', 'flux_n2b',
              'flux_hb', 'flux_o3a', 'flux_o3b', 'flux_s2a', 'flux_s2b',
              'flux_o2')

    def __init__(self):
        for key, widget in self.widgets.items():
            widget.style = self.style
            widget.layout = self.layout

        self.widgets['bulge_scale'].observe(self.compute_radius, names='value')
        self.widgets['disk_scale'].observe(self.compute_radius, names='value')
        self.widgets['bulge_fraction'].observe(self.compute_radius, names='value')
        self.widgets['axis_ratio'].observe(self.compute_radius, names='value')

        self.widgets['flux_ha'].observe(self.flux_ha_change, names='value')
        self.widgets['n2_ha_ratio'].observe(self.n2_ha_ratio_change, names='value')
        self.widgets['flux_n2a'].observe(self.flux_n2a_change, names='value')
        self.widgets['flux_n2b'].observe(self.flux_n2b_change, names='value')
        self.widgets['flux_s2a'].observe(self.flux_s2a_change, names='value')
        self.widgets['flux_s2b'].observe(self.flux_s2b_change, names='value')
        self.widgets['flux_o3a'].observe(self.flux_o3a_change, names='value')
        self.widgets['flux_o3b'].observe(self.flux_o3b_change, names='value')

        self.compute_radius()

        title = HTML("<h2>Galaxy</h2>")
        n2box = HBox([self.widgets['n2_ha_ratio'], self.widgets['flux_n2a'], self.widgets['flux_n2b']])
        s2box = HBox([self.widgets['flux_s2a'], self.widgets['flux_s2b']])
        o3box = HBox([self.widgets['flux_o3a'], self.widgets['flux_o3b']])
        hbbox = HBox([self.widgets['flux_hb']])
        o2box = HBox([self.widgets['flux_o2']])

        elements = []
        elements += [self.widgets['redshift']]

        sizebox = VBox([self.widgets['bulge_scale'], self.widgets['disk_scale'], self.widgets['bulge_fraction'], self.widgets['axis_ratio'], self.widgets['pa'],self.widgets['iso'] ])
        elements += [HTML("<b>Size</b>"), HBox([sizebox, self.widgets['half_light_radius']])]
        elements += [HTML("<b>Emission lines</b>"), self.widgets['velocity_dispersion'], self.widgets['flux_ha'], n2box,s2box,hbbox,o3box,o2box]

        self.widget = VBox(elements)

    def compute_radius(self, change=None):
        """ """
        radius = bulgy_disk_radius(
            np.array([self.widgets['bulge_scale'].value]),
            np.array([self.widgets['disk_scale'].value]),
            np.array([self.widgets['bulge_fraction'].value]),
            np.array([self.widgets['axis_ratio'].value]),
            0.5)[0]
        self.widgets['half_light_radius'].children[1].value = "%3.2f"%radius

    def n2_ha_ratio_change(self, change):
        y = self.widgets['n2_ha_ratio'].value * self.widgets['flux_ha'].value
        self.widgets['flux_n2a'].value = y / 4.
        self.widgets['flux_n2b'].value = y * 3 / 4.

    flux_ha_change = n2_ha_ratio_change

    def flux_n2b_change(self, change):
        self.widgets['flux_n2a'].value = self.widgets['flux_n2b'].value / 3.
        self.widgets['n2_ha_ratio'].value = (self.widgets['flux_n2a'].value + self.widgets['flux_n2b'].value) / self.widgets['flux_ha'].value

    def flux_n2a_change(self, change):
        self.widgets['flux_n2b'].value = self.widgets['flux_n2a'].value * 3.
        self.widgets['n2_ha_ratio'].value = (self.widgets['flux_n2a'].value + self.widgets['flux_n2b'].value) / self.widgets['flux_ha'].value

    def flux_s2b_change(self, change):
        self.widgets['flux_s2a'].value = self.widgets['flux_s2b'].value

    def flux_s2a_change(self, change):
        self.widgets['flux_s2b'].value = self.widgets['flux_s2a'].value

    def flux_o3b_change(self, change):
        self.widgets['flux_o3a'].value = self.widgets['flux_o3b'].value / 3.

    def flux_o3a_change(self, change):
        self.widgets['flux_o3b'].value = self.widgets['flux_o3a'].value * 3.

