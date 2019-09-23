Pypelid Calc
============

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bengranett/pypelidcalc/master?urlpath=/apps/notebook/pypelid-snr.ipynb)

Contact
-------
Ben Granett (https://github.com/bengranett/pypelidcalc)

Web app
-------
The calculator may be used as a web app by launching the notebook from [mybinder.org](https://mybinder.org/).
Just click the button: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bengranett/pypelidcalc/master?urlpath=/apps/notebook/pypelid-snr.ipynb)

Installation
------------

jupyter nbextension install --py widgetsnbextension --sys-prefix
jupyter nbextension enable widgetsnbextension --py --sys-prefix

jupyter nbextension install --py plotlywidget --sys-prefix
jupyter nbextension enable plotlywidget --py --sys-prefix

jupyter nbextension     enable --py --sys-prefix appmode
jupyter serverextension enable --py --sys-prefix appmode
