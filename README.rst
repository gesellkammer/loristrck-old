LORISTRCK
---------

This is the simplest wrapper possible for the partial-tracking library Loris. 
The source of the library is included as part of the project, so there is no need
to install the library independently. 

C++ Library Dependencies
------------------------

* fftw3_

.. _fftw3: http://www.fftw.org


Additional Python Module Dependencies
-------------------------------------

* Python (>= 2.7.*)
* Cython_
* NumPy
* SciPy
* sndfileio

.. _Cython: http://cython.org


Installation
------------

To build and install everything, from the root folder run:

::

    $ python setup.py install
    
Usage
-----

::
    from loristrck import analyze
    from e.sndfileio import read_sndfile
    sndfile = read_sndfile("/path/to/sndfile.wav")
    partials = analyze(sndfile.samples, sndfile.sr, resolution=50, window_width=80)
    for label, data in partials:
        print data

data will be a numpy array of shape = (numframes, 5) with the columns:

time . frequency . amplitude . phase . bandwidth

Goal
----

The main goal was as an analysis tool for the package `sndtrck`, which implements
an agnostic data structure to handle partial tracking information. So if `trckr`
is installed, it can be used as:

::
    import sndtrck
    spectrum = sndtrck.analyze_loris("/path/to/sndfile", resolution=50)
    print spectrum.chord_at(0.5)
    spectrum.plot()

Credits
-------

eduardo dot moguillansky @ gmail dot com
