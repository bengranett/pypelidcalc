from collections import OrderedDict
import gzip
import StringIO

from ipywidgets import VBox, Textarea, Button, Text

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
    layout = {'width': '500px', 'height': '500px'}

    widgets = {
        'configarea': Textarea(value='', placeholder='# paste config parameters here', description='Configuration'),
        'short': Text(description='quick code:')
    }

    def __init__(self, widget_list):
        """ """
        self.widget_list = widget_list

        self.widgets['configarea'].layout = self.layout

        button = Button(description="Load", icon='play')

        button.on_click(self.load_params)

        self.widget = VBox([self.widgets['configarea'], self.widgets['short'], button])

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

        # self.get_short()

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
                pass

            if key not in parameter_names:
                continue
            for w in self.widget_list:
                if key in w.widgets:
                    parsed_lines.append(line)
                    w.widgets[key].value = value
        param_listing = "\n".join(parsed_lines)
        # self.widgets['configarea'].value = param_listing
        self.update()

    def get_short(self):
        config = self.widgets['configarea'].value
        config = config.replace(" ", "")
        data = StringIO.StringIO()
        with gzip.GzipFile(fileobj=data, mode='w') as f:
            f.write(config)
        s = data.getvalue().encode('hex')
        self.widgets['short'].value = s



