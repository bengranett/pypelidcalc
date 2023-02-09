# -*- coding: utf-8 -*-
#!/usr/bin/python
""" This is the setup script for Pypelid which uses setuptools for automatic installation.

    Based on:
    - https://packaging.python.org/en/latest/distributing.html
    - https://github.com/pypa/sampleproject.

    Some reasources about how to write one of these that may or may not add to one's understanding:
    - http://pythonhosted.org/setuptools,
    - https://docs.python.org/2/distutils,
    - http://python-packaging.readthedocs.org,
    - https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts,
    - http://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use,
    - https://www.digitalocean.com/community/tutorials/how-to-package-and-distribute-python-applications

"""
import sys

# Always prefer setuptools over distutils
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from distutils.command.clean import clean as Clean

# To use a consistent encoding
from codecs import open
import os
import shutil

def import_install(name, pypi_name=None):
    """ """
    try:
        return __import__(name)
    except ImportError:
        import pip._internal
        try:
            if not pypi_name:
                pypi_name = name
            pip._internal.main(["install", pypi_name])
            return __import__(name)
        except:
            print("WARNING: %s not installed."%name)
            raise


numpy = import_install("numpy")
cython = import_install("cython")
cython_gsl = import_install("cython_gsl", "cythongsl")


here = os.path.abspath(os.path.dirname(__file__))

os.system("git describe --always --dirty --broken && echo version = \\\"`git describe`\\\" > pypelidcalc/version.py")
os.system("git rev-parse && echo git_revision = \\\"`git rev-parse HEAD`\\\" >> pypelidcalc/version.py")

from pypelidcalc import __version__

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


from Cython.Distutils import build_ext, Extension

include = [cython_gsl.get_include(), numpy.get_include()]
libraries = cython_gsl.get_libraries() + ['m']
library_dirs = [cython_gsl.get_library_dir()],
include_dirs = [cython_gsl.get_cython_include_dir(),numpy.get_include()]


cython_files = [
    "pypelidcalc/cutils/histogram.pyx",
    "pypelidcalc/cutils/interpolate.pyx",
    "pypelidcalc/cutils/rootfinder.pyx",
    "pypelidcalc/cutils/rng.pyx",
    "pypelidcalc/spectra/_profile_calc.pyx",
    "pypelidcalc/spectra/bulgy_disk.pyx",
    "pypelidcalc/spectra/galaxy.pyx",
    "pypelidcalc/spectra/linesim.pyx",
    "pypelidcalc/survey/psf.pyx",
]

cython_directives = {
    'embedsignature': True,
    'language_level': 3,
    'boundscheck': False,
    'wraparound': False,
    'nonecheck': False,
    'cdivision': True,
    'profile': False,
}

ext_modules = [ ]

for path in cython_files:
    name = path[:-4].replace("/",".")
    ext_modules += [Extension(name, [path], include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs, cython_directives=cython_directives)]

def unlink(path):
    if os.path.exists(path):
        print("unlink %s"%path, file=sys.stderr)
        os.unlink(path)

class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('pypelidcalc'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    for cython_ext in ['.pyx','.pxd']:
                        cython_file = str.replace(filename, extension, cython_ext)
                        if os.path.exists(os.path.join(dirpath, cython_file)):
                            unlink(os.path.join(dirpath, filename))

            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'build_ext': build_ext, 'clean': CleanCommand}


def get_requirements():
    with open(os.path.join(".","requirements.txt")) as req_file:
        reqs = [line.strip() for line in req_file]
    return reqs


setup(
    name = 'pypelidcalc',
    python_requires='==3.*',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version = __version__,

    description = 'Emission line galaxy spectra signal to noise calculator',
    long_description = long_description,

    # The project's main homepage.
    url = '',
    download_url = 'none',

    # Author details
    author = 'Ben Granett',
    author_email = 'granett@gmail.com',

    license = "GPL",


    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GPL License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='astronomy',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages = [
        'pypelidcalc', # Maybe use setuptools.find_packages() instead?
        'pypelidcalc.survey',
        'pypelidcalc.utils',
        'pypelidcalc.cutils',
        'pypelidcalc.spectra',
        'pypelidcalc.widget',
        ],

    cmdclass = cmdclass,
    include_dirs = include,
    ext_modules=ext_modules,

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],


    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # Other possible resources about this rather complicated topic:
    # - https://pip.pypa.io/en/latest/user_guide/#requirements-files,
    # - https://pip.pypa.io/en/latest/reference/pip_install/#requirements-file-format,
    # - http://setuptools.readthedocs.io/en/latest/setuptools.html#id13,
    # - http://stackoverflow.com/questions/11032125/how-can-i-make-setuptools-install-a-package-from-another-source-thats-also-avai,
    install_requires = get_requirements(),

    # Tests
    test_suite = 'nose.collector',
    tests_require = ['nose'],
)
