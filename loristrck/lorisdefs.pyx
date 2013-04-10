#cython: embedsignature=True
cimport lorisdefs as loris
from libcpp.map cimport map as stdmap
import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
import warnings
np.import_array()

ctypedef np.float64_t SAMPLE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def analyze(double[::1] samples not None, double srate, double resolution, double window_width= -1, double hop_time=-1, double freq_drift=-1, sidelobe_level=-1, amp_floor=-90):
    """
    returns a generator. it yields a tuple (label, data) for each partial, where data is
    a 2D numpy array with columns
    
    time freq amp phase bw

    samples: an array representing a mono sndfile. 
             NB: if you have a stereo array, a channel can be selected with:
             sound[:,0]  --> the "left" channel
             sound[:,1]  --> the "right" channel
    srate:   the sampling rate
    resolution: in Hz (as passed to Loris's Analyzer)
    window_width: in Hz.

    The rest of the parameters are set with sensible defaults if not given
    explicitely. 
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

    cdef double *samples0 = &(samples[0])              #<double*> np.PyArray_DATA(samples)
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
cdef np.ndarray partial_to_array(loris.Partial* p):
    cdef int numbps = p.numBreakpoints()
    cdef np.ndarray [SAMPLE_t, ndim=2] arr = np.empty((numbps, 5), dtype='float64')
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


