#cython: embedsignature=True
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport lorisdefs as loris
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
import numpy  as _np
cimport numpy as _np

_np.import_array()

ctypedef _np.float64_t SAMPLE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def analyze(double[::1] samples not None, double srate, double resolution, double window_width= -1, 
            double hop_time=-1, double freq_drift=-1, sidelobe_level=-1, amp_floor=-90):
    """
    Partial Tracking Analysis
    =========================

    Analyze the audio samples, returns a generator for each partial.
    It yields a tuple (label, data) for each partial, where:

    label --> the label (an int) of the partial
    data  --> a 2D numpy array with the columns: 

              time freq amp phase bw

    Arguments
    =========

    samples ------> an array representing a mono sndfile. 
                    >>> samples = read_aiff(sndfile)
                    NB: if you have a stereo array, a channel can be selected with:
                    >>> samples[:,0]  --> the "left" channel
                    >>> samples[:,1]  --> the "right" channel
    srate --------> the sampling rate
    resolution ---> in Hz (as passed to Loris's Analyzer). Only one partial will be found
                    within this distance. Usable values range from 30 Hz to 200 Hz.
    window_width -> in Hz. 

    The rest of the parameters are set with sensible defaults if not given explicitely.
    (a value of -1 indicates that a default value should be set)

    hop_time      : (sec) The time to move the window after each analysis. 
    freq_drift    : (Hz)  The maximum variation of frecuency between two breakpoints to be
                          considered to belong to the same partial
    sidelobe_level: (dB)  A positive dB value, indicates the shape of the Kaiser window
                          (typical value: 90 dB)
    amp_floor     : (dB)  A breakpoint with an amplitude lower than this value will not
                          be considered
    
    """
    if window_width < 0:
        window_width = resolution * 2  # original Loris behaviour
    cdef loris.Analyzer* an = new loris.Analyzer(resolution, window_width)
    if hop_time > 0:
        an.setHopTime( hop_time )
    if freq_drift > 0:
        an.setFreqDrift( freq_drift )
    if sidelobe_level > 0:
        an.setSidelobeLevel( sidelobe_level )
    an.setAmpFloor(amp_floor)

    cdef double *samples0 = &(samples[0])              #<double*> _np.PyArray_DATA(samples)
    cdef double *samples1 = &(samples[<int>(samples.size-1)]) #samples0 + <int>(samples.size - 1)
    an.analyze(samples0, samples1, srate)  

    # yield all partials
    cdef loris.PartialList partials = an.partials()
    cdef loris.PartialListIterator p_it = partials.begin()
    cdef loris.PartialListIterator p_end = partials.end()
    cdef loris.Partial partial
    cdef int n = 0
    while p_it != p_end:
        partial = deref(p_it)
        yield partial.label(), partial_to_array(&partial)
        inc(p_it)
    del an

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _np.ndarray partial_to_array(loris.Partial* p):
    cdef int numbps = p.numBreakpoints()
    cdef _np.ndarray [SAMPLE_t, ndim=2] arr = _np.empty((numbps, 5), dtype='float64')
    cdef double[:, :] a = arr
    cdef loris.Partial_Iterator it  = p.begin()
    cdef loris.Partial_Iterator end = p.end()
    cdef loris.Breakpoint bp
    cdef double time
    cdef int i = 0
    while it != end:
        bp = it.breakpoint()
        time = it.time()
        a[i, 0] = time
        a[i, 1] = bp.frequency()
        a[i, 2] = bp.amplitude()
        a[i, 3] = bp.phase()
        a[i, 4] = bp.bandwidth()
        i += 1
        inc(it)
    return arr

def read_sdif(sdiffile):
    """
    Read the SDIF file

    sdiffile: (str) The path to a SDIF file

    Returns
    =======

    a generator yielding a tuple (label, data) for each partial in the SDIF file, where:

    label: the label (an int) of the partial
    data : a 2D numpy array with the columns --> time freq amp phase bw

    """
    cdef loris.SdifFile* sdif = new loris.SdifFile(string(<char*>sdiffile))
    cdef loris.PartialList partials = sdif.partials()

    # yield all partials
    cdef loris.PartialListIterator p_it = partials.begin()
    cdef loris.PartialListIterator p_end = partials.end()
    cdef loris.Partial partial
    cdef int n = 0
    while p_it != p_end:
        partial = deref(p_it)
        yield partial.label(), partial_to_array(&partial)
        inc(p_it)
    del sdif

def read_aiff(path):
    """
    Read a mono AIFF file (Loris does not read stereo files)

    path: (str) The path to the soundfile (.aif or .aiff)

    -> Raises ValueError if the soundfile is not mono

    Returns
    =======

    A tuple (audiodata, samplerate) with

    audiodata : a 1D numpy array of type double
    """
    cdef loris.AiffFile* f = new loris.AiffFile(string(<char*>path))
    cdef vector[double] samples = f.samples()
    cdef double[:] mono
    cdef int i, numFrames
    cdef vector[double].iterator it
    cdef int channels = f.numChannels()
    cdef double samplerate = f.sampleRate()
    if channels == 1:
        numFrames = f.numFrames()
        mono = _np.empty((numFrames,), dtype='float64')
        it = samples.begin()
        while i < numFrames:
            mono[i] = deref(it)
            i += 1
            inc(it)
        return (mono, samplerate)
    else:
        raise ValueError("attempting to read a multi-channel (>1) AIFF file!")