#cython: embedsignature=True
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport lorisdefs as loris
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as _np
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
    cdef loris.Breakpoint *bp
    cdef double time
    cdef int i = 0
    while it != end:
        bp = &(it.breakpoint())
        a[i, 0] = it.time()
        a[i, 1] = bp.frequency()
        a[i, 2] = bp.amplitude()
        a[i, 3] = bp.phase()
        a[i, 4] = bp.bandwidth()
        i += 1
        inc(it)
    return arr

@cython.boundscheck(False)
@cython.wraparound(False)
cdef loris.Partial* array_to_partial(_np.ndarray[SAMPLE_t, ndim=2] a, double fadetime=0.0):
    cdef loris.Partial *p = new loris.Partial()
    cdef int numbps = len(a)
    cdef loris.Breakpoint *bp
    
    if fadetime > 0 and a[0, 2] > 0:
        bp = new loris.Breakpoint()
        bp.setFrequency(a[0, 1])
        bp.setAmplitude(0)
        bp.setPhase(a[0, 3])
        bp.setBandwidth(a[0, 4])
        p.insert(a[0, 0] - fadetime, deref(bp))

    # each row in a is (time, freq, amp, phase, bw)
    for i in range(numbps):
        bp = new loris.Breakpoint()
        bp.setFrequency(a[i, 1])
        bp.setAmplitude(a[i, 2])
        bp.setPhase(a[i, 3])
        bp.setBandwidth(a[i, 4])
        p.insert(a[i, 0], deref(bp))

    if fadetime > 0 and a[numbps-1, 2] > 0:
        bp = new loris.Breakpoint()
        bp.setFrequency(a[numbps-1, 1])
        bp.setAmplitude(0)
        bp.setPhase(a[numbps-1, 3])
        bp.setBandwidth(a[numbps-1, 4])
        p.insert(a[numbps-1, 0] + fadetime, deref(bp))

    return p

@cython.boundscheck(False)
@cython.wraparound(False)
cdef loris.Partial* array_to_partial2(_np.ndarray[SAMPLE_t, ndim=2] a, double fadetime=0.0):

    cdef int numbps = len(a)
    cdef loris.Partial *p = new loris.Partial()
    cdef loris.Breakpoint *bp
    cdef loris.Breakpoint *bp2
    cdef double amp, t
    cdef loris.Breakpoint b
    # each row in a is (time, freq, amp, phase, bw)
    for i in range(numbps):
        bp = new loris.Breakpoint()
        bp.setFrequency(a[i, 1])
        bp.setAmplitude(a[i, 2])
        bp.setPhase(a[i, 3])
        bp.setBandwidth(a[i, 4])
        p.insert(a[i, 0], deref(bp))

    if fadetime > 0:
        # fadetime only takes effect if the partial begins or ends with non-zero amp
        bp = &(p.first())
        t = p.startTime() - fadetime

        amp = bp.amplitude()
        if amp > 0:
            bp2 = new loris.Breakpoint()
            bp2.setFrequency(bp.frequency())
            bp2.setAmplitude(0)
            bp2.setPhase(p.phaseAt(t))
            bp2.setBandwidth(bp.bandwidth())
            p.insert(t, deref(bp2))

        bp = &(p.last())
        t = p.endTime() + fadetime

        amp = bp.amplitude()
        if amp > 0:
            bp2 = new loris.Breakpoint()
            bp2.setFrequency(bp.frequency())
            bp2.setAmplitude(0)
            bp2.setPhase(p.phaseAt(t))
            bp2.setBandwidth(bp.bandwidth())
            p.insert(t, deref(bp2))

    return p

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
    cdef loris.SdifFile* sdif
    cdef loris.PartialList partials
    cdef string filename = string(<char*>sdiffile)
    sdif = new loris.SdifFile(filename)
    partials = sdif.partials()
    # yield all partials
    cdef loris.PartialListIterator p_it = partials.begin()
    cdef loris.PartialListIterator p_end = partials.end()
    cdef loris.Partial partial
    cdef int n = 0
    while p_it != p_end:
        partial = deref(p_it)
        yield (partial.label(), partial_to_array(&partial))
        inc(p_it)
    del sdif

def write_sdif(outfile, partials, labels=None, rbep=True, double fadetime=0):
    """
    Write a list of partials in the sdif 
    partials: a seq. of matrices where matrix is a 2D numpy arrays of the format [time freq amp phase bw]
    labels: a seq. if integer labels
    rbep: if True, use RBEP format, otherwise, 1TRC

    NB: The 1TRC format forces resampling
    """
    cdef loris.PartialList *partial_list = _partials_from_data(partials, fadetime)
    cdef loris.SdifFile* sdiffile = new loris.SdifFile(partial_list.begin(), partial_list.end())
    print("saving to sdif")
    cdef string filename = string(<char*>outfile)
    cdef int use_rbep = int(rbep)
    if labels is not None:
        _partials_set_labels(partial_list, labels)
    with nogil:
        if use_rbep:
            sdiffile.write(filename)
        else:
            sdiffile.write1TRC(filename)
    del partials
    del sdiffile

cdef void _partials_set_labels(loris.PartialList *partial_list, labels):
    cdef loris.PartialListIterator p_it = partial_list.begin()
    cdef loris.PartialListIterator p_end = partial_list.end()
    cdef loris.Partial partial
    for label in labels:
        partial = deref(p_it)
        partial.setLabel(label)
        inc(p_it)
        if p_it == p_end:
            break

cdef loris.PartialList* _partials_from_data(data, double fadetime=0):
    """
    data: a seq. where matrix is a 2D array of type 
          [time, freq, amp, phase, bw]

    NB: to set the labels of the partials, call _partials_set_labels
    """
    cdef loris.PartialList *partials = new loris.PartialList()
    cdef loris.Partial *partial
    cdef int i = 0
    cdef int label
    for matrix in data:
        partial = array_to_partial(matrix, fadetime)
        partials.push_back(deref(partial))
        i += 1
    return partials

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

cdef object _partials_timespan(loris.PartialList * partials):
    cdef loris.PartialListIterator it = partials.begin()
    cdef loris.PartialListIterator end = partials.end()
    cdef loris.Partial partial = deref(it)
    cdef double tmin = partial.startTime()
    cdef double tmax = partial.endTime()
    while it != end:
        partial = deref(it)
        tmin = min(tmin, partial.startTime())
        tmax = max(tmax, partial.endTime())
        inc(it)
    return tmin, tmax

def synthesize(data, samplerate=48000, time_selection=None):
    """
    data: a seq. of 2D matrices, each matrix represents a partial
              Each row is a breakpoint of the form [time freq amp phase bw]
    time_selection: if given, a tuple (start_time, end_time).
                    (0.5, None) --> synthesize from 0.5 to end of spectrum
                    (None, 10)  --> synthesize from beginning to time=10
    """

    if time_selection is not None:
        t0, t1 = time_selection
    cdef loris.PartialList *partials = _partials_from_data(data)
    tmin, tmax = _partials_timespan(partials)
    if t0 is None:
        t0 = tmin
    if t1 is None:
        t1 = tmax
    del partials
    return 