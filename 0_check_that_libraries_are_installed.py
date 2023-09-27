# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:03:24 2022

@author: RockLee
"""

from __future__ import print_function
from distutils.version import LooseVersion as Version
import sys

try:
    import curses
    curses.setupterm()
    assert curses.tigetnum("colors") > 2
    OK = "\x1b[1;%dm[ OK ]\x1b[0m" % (30 + curses.COLOR_GREEN)
    FAIL = "\x1b[1;%dm[FAIL]\x1b[0m" % (30 + curses.COLOR_RED)
except:
    OK = '[ OK ]'
    FAIL = '[FAIL]'

try:
    import importlib
except ImportError:
    print(FAIL, "Python version 3.10.8 is required,"
                " but %s is installed." % sys.version)

def import_version(pkg, min_ver):
    mod = None
    try:
        mod = importlib.import_module(pkg)
        if pkg in {'PIL'}:
            ver = mod.VERSION
        else:
            ver = mod.__version__
        if Version(ver) < min_ver:
            print(FAIL, "%s version %s or higher required, but %s installed."
                  % (lib, min_ver, ver))
        else:
            print(OK, '%s version %s' % (pkg, ver))
    except ImportError as imp_err_msg:
        print(FAIL, 'Error in importing %s: %s' % (pkg, imp_err_msg))
    except AttributeError as att_err_msg:
        print(FAIL, 'Error in reading attribute of %s: %s' % (pkg, att_err_msg))
    return mod

# first check the python version
print('Using python in', sys.prefix)
print(sys.version)
pyversion = Version(sys.version)
if pyversion >= "3":
    if pyversion < "3.10.8":
        print(FAIL, "Python version > 3.10.8 is required,"
                    " but %s is installed.\n" % sys.version)
elif pyversion < "3":
    print(FAIL, "Python version > 3.10.8 is required,"
                " but %s is installed.\n" % sys.version)
else:
    print(FAIL, "Unknown Python version: %s\n" % sys.version)

requirements = {'numpy': '1.18.0',
                'pandas': '1.0.0',
                'matplotlib': '3.5.1',
                'seaborn': '0.11.2',
                'sklearn': '1.1.2',
                'scipy': '1.7.3'}

# now check the dependencies
for lib, required_version in list(requirements.items()):
    import_version(lib, required_version)