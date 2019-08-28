from collections import OrderedDict
import gzip
import StringIO

from ipywidgets import VBox, HTML, Textarea, Button, Text

import traitlets

import instrument_widget, foreground_widget, galaxy_widget, analysis_widget, survey_widget


objs = (
    galaxy_widget.Galaxy,
    survey_widget.Survey,
    foreground_widget.Foreground,
    instrument_widget.Instrument,
    analysis_widget.Analysis,
)

parameter_names = []

for w in objs:
    for key in w.params:
        parameter_names.append(key)



class Config(object):
    style = {'description_width': '200px'}
    layout = {'width': '500px', 'height': '200px'}

    widgets = {
        'configarea': Textarea(value='', placeholder='# paste config parameters here', description='Edit configuration list:'),
    }

    def __init__(self, widget_list):
        """ """
        self.widget_list = widget_list

        self.widgets['configarea'].layout = self.layout
        self.widgets['configarea'].style = self.style

        button = Button(description="Apply")
        button.style.button_color = 'lightgreen'

        button.on_click(self.load_params)

        self.widget = VBox([self.widgets['configarea'], button])

        self.update()

    def update(self):
        """ """
        params = OrderedDict()

        for w in self.widget_list:
            for key in parameter_names:
                if key in w.widgets:
                    try:
                        params[key] = w.widgets[key].value
                    except AttributeError:
                        print "error", key
                        continue

        param_listing = ""
        for key, value in params.items():
            param_listing += "%s = %s\n"%(key, value)
        self.widgets['configarea'].value = param_listing

    def load_params(self, button):
        """ """
        parsed_lines = []
        data = self.widgets['configarea'].value
        for line in data.split("\n"):
            line = line.strip()
            if line == "":
                continue
            if line.startswith("#"):
                parsed_lines.append(line)
                continue
            try:
                key, value = line.split("=")
            except:
                print "error",line, line.split("=")
            key = key.strip()
            value = value.strip()

            try:
                value = float(value)
            except ValueError:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    pass

            if key not in parameter_names:
                continue
            for w in self.widget_list:
                if key in w.widgets:
                    parsed_lines.append(line)
                    try:
                        w.widgets[key].value = value
                    except traitlets.TraitError:
                        print "error setting",key,value
        param_listing = "\n".join(parsed_lines)
        # self.widgets['configarea'].value = param_listing
        self.update()


