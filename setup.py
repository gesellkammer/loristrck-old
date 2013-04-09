'''
DESCRIPTION
'''
import os
import glob
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# ----------------------------------------------
# monkey-patch for parallel compilation
def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build =  self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    # parallel code
    N=2 # number of parallel compilations
    import multiprocessing.pool
    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).imap(_single_compile,objects))
    return objects
import distutils.ccompiler
distutils.ccompiler.CCompiler.compile=parallelCCompile

# -----------------------------------------------------------------------------
# Global
# -----------------------------------------------------------------------------

# detect platform
platform = os.uname()[0] if hasattr(os, 'uname') else 'Windows'

# get numpy include directory
try:
    import numpy
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
except ImportError:
    print 'Error: Numpy was not found.'
    exit(1)

macros = []
link_args = []
include_dirs = ['ltrckr', 'src/ltrckr',
                'src/loris', numpy_include, '/usr/local/include', 
                '/src/fftw',    # the path of the directory where fftw was unzipped
                '/GSL/include'  # the path where the GSL include files are
                ]
libs = ['m', 'fftw3', 'gsl', 'gslcblas']
compile_args = ['-DMERSENNE_TWISTER', '-DHAVE_FFTW3_H', "-march=i686"]
library_dirs = [
    '/GSL/bin', '/GSL/lib',   # the path to the GSL .dll(s)
    '/src/fftw'               # the path to the fftw .dll(s)
    ]

if platform == 'Windows':
    print "NB: make sure that the GSL and FFTW dlls are in the windows PATH"

sources = []

# -----------------------------------------------------------------------------
# Loris
# -----------------------------------------------------------------------------
loris_base = os.path.join(*'src loris src'.split())
loris_sources = glob.glob(os.path.join(loris_base, '*.C'))
loris_exclude = []
#loris_exclude += glob.glob(os.path.join(loris_base, 'loris*.C'))
loris_exclude += [os.path.join(loris_base, filename) for filename in  \
    (
        "ImportLemur.C",
        "Dilator.C",
        "Morpher.C",
        "SpectralSurface.C",
        "lorisNonObj_pi.C",
        "Channelizer.C",
        "Distiller.C",
        "PartialUtils.C",
        "lorisUtilities_pi.C",
        "lorisPartialList_pi.C",
        "lorisAnalyzer_pi.C",
        "lorisBpEnvelope_pi.C",
        "Harmonifier.C",
        "Collator.C",
        "lorisException_pi.C"
    )]   


loris_sources = list(set(loris_sources) - set(loris_exclude))
sources.extend(loris_sources)

# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------

# base = Extension(
#     'ltrckr.base',
#     sources=['ltrckr/base.pyx',
#              'src/ltrckr/base.cpp',
#              'src/ltrckr/exceptions.cpp'],
#     include_dirs=include_dirs,
#     library_dirs=library_dirs,
#     language='c++'
# )

# -----------------------------------------------------------------------------
# Loris
# -----------------------------------------------------------------------------
loris = Extension(
    'ltrckr.lorisdefs',
    sources=sources + ['ltrckr/lorisdefs.pyx'],
    include_dirs=include_dirs,
    libraries=libs,
    library_dirs=library_dirs,
    extra_compile_args=compile_args,
    language='c++'
)

# -----------------------------------------------------------------------------
# Partial Tracking
# -----------------------------------------------------------------------------
# partial_tracking = Extension(
#     'ltrckr.partial_tracking',
#     sources=sources + ['ltrckr/partial_tracking.pyx',
#                        'src/ltrckr/partial_tracking.cpp',
#                        'src/ltrckr/base.cpp',
#                        'src/ltrckr/exceptions.cpp'],
#     libraries=libs,
#     extra_compile_args=compile_args,
#     include_dirs=include_dirs,
#     library_dirs=library_dirs,
#     language='c++'
# )

# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------
# synthesis = Extension(
#     'ltrckr.synthesis',
#     sources=sources + ['ltrckr/synthesis.pyx',
#                        'src/ltrckr/synthesis.cpp',
#                        'src/ltrckr/base.cpp',
#                        'src/ltrckr/exceptions.cpp'],
#     libraries=libs,
#     extra_compile_args=compile_args,
#     include_dirs=include_dirs,
#     library_dirs=library_dirs,
#     language='c++'
# )

# -----------------------------------------------------------------------------
# Residual
# -----------------------------------------------------------------------------
# residual = Extension(
#     'ltrckr.residual',
#     sources=sources + ['ltrckr/residual.pyx',
#                        'src/ltrckr/peak_detection.cpp',
#                        'src/ltrckr/partial_tracking.cpp',
#                        'src/ltrckr/synthesis.cpp',
#                        'src/ltrckr/residual.cpp',
#                        'src/ltrckr/base.cpp',
#                        'src/ltrckr/exceptions.cpp'],
#     libraries=libs,
#     extra_compile_args=compile_args,
#     include_dirs=include_dirs,
#     library_dirs=library_dirs,
#     language='c++'
# )

# -----------------------------------------------------------------------------
# Package
# -----------------------------------------------------------------------------
doc_lines = __doc__.split('\n')

setup(
    name='ltrckr',
    description=doc_lines[0],
    long_description='\n'.join(doc_lines[2:]),
    url='http://ltrckrsound.sourceforge.net',
    download_url='http://ltrckrsound.sourceforge.net',
    license='GPL',
    author='John Glover',
    author_email='j@johnglover.net',
    platforms=['Linux', 'Mac OS-X', 'Unix'],
    version='0.3',
    ext_modules=[loris],
    cmdclass={'build_ext': build_ext},
    packages=['ltrckr']
)
