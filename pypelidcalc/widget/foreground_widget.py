from ipywidgets import VBox, BoundedFloatText


class Foreground(object):
    style = {'description_width': '200px'}
    layout = {'width': '300px'}

    widgets = {
        'foreground': BoundedFloatText(value=1, min=0, max=1e6, step=0.1, description='foreground counts (elec/s/pix)'),
    }

    def __init__(self):
        """ """
        for key, widget in self.widgets.items():
            widget.style = self.style
            widget.layout = self.layout

        self.widget = VBox([self.widgets['foreground']])
