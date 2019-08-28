from ipywidgets import HTML, VBox, BoundedFloatText, BoundedIntText, Dropdown


configurations = {
    'Euclid Wide':{'exp_time': 565, 'nexp_red': 4, 'nexp_blue': 0},
    'Euclid Deep':{'exp_time': 565, 'nexp_red': 100, 'nexp_blue': 100},
    'HST': {'exp_time': 5700, 'nexp_red': 2, 'nexp_blue':0},
    'custom': {}
}


class Survey(object):
    style = {'description_width': '200px'}
    layout = {'width': '350px'}

    widgets = {
        'config': Dropdown(options=configurations.keys(), description='Configurations:'),
        'exp_time': BoundedFloatText(value=565, min=0, max=1e6, step=5, description='Exposure time (sec)'),
        'nexp_red': BoundedIntText(value=4, min=0, max=1000, step=1, description='Number of red exposures'),
        'nexp_blue': BoundedIntText(value=0, min=0, max=1000, step=1, description='Number of blue exposures'),

    }

    params = ('exp_time', 'nexp_red', 'nexp_blue')

    def __init__(self):
        """ """
        self._set_custom = True
        for key, widget in self.widgets.items():
            widget.style = self.style
            widget.layout = self.layout


        self.update(**configurations[self.widgets['config'].value])
        self.widgets['config'].observe(self.change_config, names='value')

        for key, widget in self.widgets.items():
            if key in ['config']:
                continue
            widget.observe(self.modify, names='value')


        elements = [self.widgets['config'], HTML('<b>Exposure</b>'), self.widgets['exp_time'], self.widgets['nexp_red'], self.widgets['nexp_blue']]

        self.widget = VBox(elements)

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
