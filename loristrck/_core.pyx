# cython: embedsignature=True
# cython: boundscheck=False
# cython: wraparound=False

import warnings
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport lorisdefs as loris
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as _np
cimport numpy as _np

_np.import_array()

CONFIG = {
    'debug': False
}

ctypedef _np.float64_t SAMPLE_t


def analyze(double[::1] samples not None, double sr, double resolution, double windowsize= -1, 
            double hoptime =-1, double freqdrift =-1, sidelobe=-1, ampfloor=-90):

    """
    Partial Tracking Analysis
    =========================

    Analyze the audio samples, returns a generator for each partial.
    Returns a list of 2D numpy arrays, where each array represent a partial with
    columns: time, freq, amplitude, phase, bandwidth

    Arguments
    =========

    samples ------> an array representing a mono sndfile. 
                    >>> samples = read_aiff(sndfile)
                    NB: if you have a stereo array, a channel can be selected with:
                    >>> samples[:,0].copy()  --> the "left" channel 
                    (.copy is needed, because we need a contiguous array)
    sr ----------> the sampling rate
    resolution --> in Hz (as passed to Loris's Analyzer). Only one partial will be found
                    within this distance. Usable values range from 30 Hz to 200 Hz.
    windowsize --> in Hz. If not given, a default value is calculated 

    The rest of the parameters are set with sensible defaults if not given explicitely.
    (a value of -1 indicates that a default value should be set)

    hoptime       : (sec) The time to move the window after each analysis. 
    freqdrift     : (Hz)  The maximum variation of frecuency between two breakpoints to be
                          considered to belong to the same partial. A sensible value is
                          between 1/2 to 3/4 of resolution
    sidelobe: (dB)  A positive dB value, indicates the shape of the Kaiser window
                          (typical value: 90 dB)
    ampfloor     : (dB)  A breakpoint with an amplitude lower than this value will not
                          be considered
    
    """
    if windowsize < 0:
        windowsize = resolution * 2  # original Loris behaviour
    cdef loris.Analyzer* an = new loris.Analyzer(resolution, windowsize)
    if hoptime > 0:
        an.setHopTime(hoptime)
    if freqdrift > 0:
        an.setFreqDrift(freqdrift)
    if sidelobe > 0:
        an.setSidelobeLevel( sidelobe )
    an.setAmpFloor(ampfloor)

    cdef double *samples0 = &(samples[0])              #<double*> _np.PyArray_DATA(samples)
    cdef double *samples1 = &(samples[<int>(samples.size-1)]) #samples0 + <int>(samples.size - 1)
    an.analyze(samples0, samples1, sr)  

    # yield all partials
    cdef loris.PartialList partials = an.partials()
    cdef loris.PartialListIterator p_it = partials.begin()
    cdef loris.PartialListIterator p_end = partials.end()
    cdef loris.Partial partial
    cdef list out = []
    while p_it != p_end:
        partial = deref(p_it)
        out.append(partial_to_array(&partial))
        inc(p_it)
    del an
    return out


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


cdef loris.Partial* newpartial_from_array(_np.ndarray[SAMPLE_t, ndim=2] a, double fadetime=0.0):
    cdef loris.Partial *p = new loris.Partial()
    cdef int numbps = len(a)
    cdef loris.Breakpoint *bp    
    cdef double t
    if fadetime > 0 and a[0, 2] > 0:
        # we don't care here about Partials with negative time,
        # since we only insert breakpoints with non-negative times
        # We DO make sure that for any partial its first breakpoint
        # has an amplitude of 0 if fadetime was positive
        if a[0, 0] == 0:
            a[0, 2] = 0
        else:
            bp = new loris.Breakpoint()
            bp.setFrequency(a[0, 1])
            bp.setAmplitude(0)
            bp.setPhase(a[0, 3])
            bp.setBandwidth(a[0, 4])
            p.insert(max(0, a[0, 0] - fadetime), deref(bp))
    # each row in a is (time, freq, amp, phase, bw)
    for i in range(numbps):
        t = a[i, 0]
        if t >= 0:
            bp = new loris.Breakpoint()
            bp.setFrequency(a[i, 1])
            bp.setAmplitude(a[i, 2])
            bp.setPhase(a[i, 3])
            bp.setBandwidth(a[i, 4])
            p.insert(t, deref(bp))
    if fadetime > 0 and a[numbps-1, 2] > 0:
        bp = new loris.Breakpoint()
        bp.setFrequency(a[numbps-1, 1])
        bp.setAmplitude(0)
        bp.setPhase(a[numbps-1, 3])
        bp.setBandwidth(a[numbps-1, 4])
        p.insert(a[numbps-1, 0] + fadetime, deref(bp))
    return p


def read_sdif(str sdiffile):
    """
    Read the SDIF file

    sdiffile: (str) The path to a SDIF file

    Returns
    =======

    (list of partialdata, labels)

    Partialdata is a a list of 2D matrices. Each matrix is a partial, 
    where each row is a breakpoint of the form (time, freq, amp, phase, bw)

    labels: a list of the labels for each partial
    """
    cdef loris.SdifFile* sdif
    cdef loris.PartialList partials
    cdef bytes path = bytes(sdiffile)
    cdef string filename = string(<char*>path)
    sdif = new loris.SdifFile(filename)
    partials = sdif.partials()
    cdef loris.PartialListIterator p_it = partials.begin()
    cdef loris.PartialListIterator p_end = partials.end()
    cdef loris.Partial partial
    cdef list matrices = []
    cdef list labels = []
    while p_it != p_end:
        partial = deref(p_it)
        matrices.append(partial_to_array(&partial))
        labels.append(partial.label())
        inc(p_it)
    del sdif
    return (matrices, labels)


def _isiterable(seq):
    return hasattr(seq, '__iter__') and not isinstance(seq, (str, bytes))


cdef class PartialW:
    cdef loris.Partial *thisptr
    def __dealloc__(self):
        del self.thisptr

cdef PartialW newPartialW(loris.Partial* p):
    cdef PartialW out = PartialW()
    out.thisptr = p
    return out


def write_sdif(partials, str outfile, labels=None, rbep=True, double fadetime=0):
    """
    Write a list of partials in the sdif 
    
    partials: a seq. of 2D arrays with columns [time freq amp phase bw]
    outfile: the path of the sdif file
    labels: a seq. of integer labels, or None to skip saving labels
    rbep: if True, use RBEP format, otherwise, 1TRC

    NB: The 1TRC format forces resampling
    """
    assert _isiterable(partials)
    cdef list refs = []
    cdef loris.PartialList *partial_list = _partials_from_data(partials, refs, fadetime)
    cdef int DEBUG = CONFIG['debug']
    if DEBUG: print("Converted to PartialList")
    cdef loris.SdifFile* sdiffile = new loris.SdifFile(partial_list.begin(), partial_list.end())
    cdef bytes b_outfile = bytes(outfile)
    cdef string filename = string(<char*>b_outfile)
    cdef int use_rbep = int(rbep)
    if labels is not None:
        if DEBUG: print("Setting Labels")
        assert _isiterable(labels)
        _partials_set_labels(partial_list, labels)
    if DEBUG: print("Writing SDIF")
    with nogil:
        if use_rbep:
            sdiffile.write(filename)
        else:
            sdiffile.write1TRC(filename)
    if DEBUG: print("Finished writing SDIF")
    del sdiffile
    destroy_partiallist(partial_list, refs)
    

cdef void destroy_partiallist(loris.PartialList *partials, list refs):
    """
    refs: a list of PartialW, as filled by _partials_from_data
    """
    cdef loris.Partial *partial
    while refs:
        refs.pop()
    partials.clear()
    del partials
    

cdef void _partials_set_labels(loris.PartialList *partial_list, labels):
    cdef loris.PartialListIterator p_it = partial_list.begin()
    cdef loris.PartialListIterator p_end = partial_list.end()
    cdef loris.Partial partial
    for label in labels:
        assert isinstance(label, (int, float))
        partial = deref(p_it)
        partial.setLabel(int(label))
        inc(p_it)
        if p_it == p_end:
            break


cdef loris.PartialList* _partials_from_data(dataseq, list refs, double fadetime=0):
    """
    dataseq: a seq. of 2D double arrays, each array represents a partial
    refs: an empty list. It will be populated with references to the 
          created partials, wrapped as PartialW. You need these
          to destroy the PartialList

    NB: to set the labels of the partials, call _partials_set_labels
    """
    cdef loris.PartialList *partials = new loris.PartialList()
    cdef loris.Partial *partial
    cdef int i = 0
    cdef int label
    for matrix in dataseq:
        partial = newpartial_from_array(matrix, fadetime)
        partials.push_back(deref(partial))
        i += 1
        refs.append(newPartialW(partial))
    return partials


def read_aiff(path):
    """
    Read a mono AIFF file (Loris does not read stereo files)

    path: (str) The path to the soundfile (.aif or .aiff)

    NB: Raises ValueError if the soundfile is not mono

    --> A tuple (audiodata, samplerate)

    audiodata : 1D numpy array of type double, holding the samples
    """
    cdef loris.AiffFile* aiff = new loris.AiffFile(string(<char*>path))
    cdef vector[double] samples = aiff.samples()
    cdef double[:] mono
    cdef vector[double].iterator it
    cdef int channels = aiff.numChannels()
    cdef double samplerate = aiff.sampleRate()
    if channels != 1:
        raise ValueError("attempting to read a multi-channel (>1) AIFF file!")
    cdef int numFrames = aiff.numFrames()
    mono = _np.empty((numFrames,), dtype='float64')
    it = samples.begin()
    cdef int i = 0
    while i < numFrames:
        mono[i] = deref(it)
        i += 1
        inc(it)
    del aiff
    return (mono, samplerate)

        
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


#def synthesize_old(dataseq, int samplerate, float fadetime=-1):
#    """
#    dataseq:    a seq. of 2D matrices, each matrix represents a partial
#                Each row is a breakpoint of the form [time freq amp phase bw]
    
#    samplerate: the samplerate of the synthesized samples (Hz)
    
#    fadetime:   to avoid clicks, partials not ending in 0 amp should be faded
#                If negative, a sensible default is used (seconds)

#    --> samples: numpy 1D array of doubles holding the samples
#    """
#    if fadetime < 0:
#        fadetime = max(0.001, 32.0/samplerate)
#    cdef list refs = []
#    cdef loris.PartialList *partials = _partials_from_data(dataseq, refs, fadetime)
#    t0, t1 = _partials_timespan(partials)
#    cdef int numsamples = int(t1 * samplerate)+1
#    cdef vector[double] bufvector 
#    bufvector.reserve(numsamples)
#    for i in range(numsamples):
#        bufvector.push_back(0)
#    cdef loris.Synthesizer *synthesizer = new loris.Synthesizer(samplerate, bufvector, fadetime)
#    cdef loris.Partial *lorispartial
#    for matrix in dataseq:
#        lorispartial = newpartial_from_array(matrix, fadetime)
#        synthesizer.synthesize(lorispartial[0])
#        del lorispartial
#    cdef _np.ndarray [SAMPLE_t, ndim=1] bufnumpy = _np.zeros((numsamples,), dtype='float64')
#    cdef int firstsample = int(t0 * samplerate)
#    for i in range(numsamples):
#        bufnumpy[i] = bufvector[firstsample+i]
#    del synthesizer
#    destroy_partiallist(partials, refs)
#    return bufnumpy


def synthesize(dataseq, int samplerate, float fadetime=-1):
    """
    dataseq:    a seq. of 2D matrices, each matrix represents a partial
                Each row is a breakpoint of the form [time freq amp phase bw]
    
    samplerate: the samplerate of the synthesized samples (Hz)
    
    fadetime:   to avoid clicks, partials not ending in 0 amp should be faded
                If negative, a sensible default is used (seconds)

    --> samples: numpy 1D array of doubles holding the samples
    """
    if fadetime < 0:
        fadetime = max(0.001, 32.0/samplerate)
    cdef list matrices = list(dataseq)
    cdef float t0 = min(m[:,0][0] for m in matrices)
    cdef float t1 = max(m[:,0][-1] for m in matrices)
    cdef int numsamples = int(t1 * samplerate)+1
    cdef vector[double] bufvector 
    bufvector.reserve(numsamples)
    for i in range(numsamples):
        bufvector.push_back(0)
    cdef loris.Synthesizer *synthesizer = new loris.Synthesizer(samplerate, bufvector, fadetime)
    cdef loris.Partial *lorispartial
    # cdef _np.ndarray [SAMPLE_t, ndim=2] matrix
    for matrix in matrices:
        if matrix[0, 0] < 0:
            warnings.warn("synthesize: Partial with negative times found, skipping")
            continue
        lorispartial = newpartial_from_array(matrix, fadetime)
        synthesizer.synthesize(lorispartial[0])
        del lorispartial
    cdef _np.ndarray [SAMPLE_t, ndim=1] bufnumpy = _np.zeros((numsamples,), dtype='float64')
    cdef int firstsample = int(t0 * samplerate)
    for i in range(numsamples):
        bufnumpy[i] = bufvector[firstsample+i]
    del synthesizer
    return bufnumpy