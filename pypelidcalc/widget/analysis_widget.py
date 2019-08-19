import os

import pandas
from IPython.display import display, clear_output
from ipywidgets import Label, HTML, VBox, BoundedFloatText, BoundedIntText, Dropdown, Output


TEMPLATE_DIR = 'templates'

def get_template_files(dir=TEMPLATE_DIR):
    filelist = []
    if not os.path.exists(dir):
        return filelist
    for filename in os.listdir(dir):
        if filename.endswith("txt"):
            filelist.append(filename)
    return filelist

class Analysis(object):
    style = {'description_width': '200px'}
    layout = {'width': '400px'}

    widgets = {
        'nloops': BoundedIntText(value=1000, min=1, max=1e6, step=1, description='Numer of realizations'),
        'zmeas_template_file': Dropdown(options=get_template_files(), description='Spec templates file'),
        'extraction_sigma': BoundedFloatText(value=2, min=0, max=100, step=0.1, description='Extraction kernel width (pixels)'),
        'zmin':  BoundedFloatText(value=0, min=0, max=10, step=0.1, description='Redshift grid min'),
        'zmax':  BoundedFloatText(value=3, min=0, max=10, step=0.1, description='Redshift grid max'),
        'zstep':  BoundedFloatText(value=0.001, min=0, max=0.01, step=0.0005, description='Redshift grid step'),

        'output': Output(layout={'width':'800px'}),
    }

    params = ('nloops', 'zmeas_template_file', 'extraction_sigma', 'zmin', 'zmax', 'zstep')

    def __init__(self):
        """ """
        for key, widget in self.widgets.items():
            widget.style = self.style
            widget.layout = self.layout
        self.widgets['output'].layout={'width':'800px'}

        self.widgets['zmeas_template_file'].observe(self.show_template_table, names='value')

        elements = []
        elements += [HTML('<b>Statistics</b>'), self.widgets['nloops']]
        elements += [HTML('<b>Extraction</b>'), self.widgets['extraction_sigma']]
        elements += [HTML('<b>Redshift measurement</b>'), self.widgets['zmeas_template_file'],
                    self.widgets['zmin'], self.widgets['zmax'], self.widgets['zstep']]


        bot = VBox([HTML('<b>Spec templates table</b>'), self.widgets['output']], layout={'width':'800px', 'display':'flex'})

        top = VBox(elements, layout={'width':'800px', 'display':'flex'})

        self.widget = VBox([top, bot], layout={'width':'800px'})

        self.show_template_table()

    def show_template_table(self, change=None):
        with self.widgets['output']:
            clear_output()
            display(Label(self.widgets['zmeas_template_file'].value))
            self.template_path = os.path.join(TEMPLATE_DIR, self.widgets['zmeas_template_file'].value)
            table = pandas.read_csv(self.template_path, sep=",")
            display(table)