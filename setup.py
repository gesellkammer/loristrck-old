'''
LORISTRCK: a basic wrapper around the partial-tracking library Loris

This is the simplest wrapper possible for the partial-tracking library Loris. 
The source of the library is included as part of the project --there is no need
to install the library independently. 

The main goal was as an analysis tool for the package `sndtrck`, which implements
an agnostic data structure to handle partial tracking information

Dependencies: 

* fftw3

'''
import os
import sys
import glob
from setuptools import setup, Extension
try:
    from Cython.Distutils import build_ext
    import numpy
except ImportError:
    setup(
        install_requires=[
            'cython>=0.19',
            'numpy>1.5'
        ]
    )
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        print "Cython is necessary to build this package."
        print "An attempt to install Cython just failed."
        print "Please install it on your own and try again."
        sys.exit()

PARALLEL = False

# ----------------------------------------------
# monkey-patch for parallel compilation
# ----------------------------------------------
def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build =  self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    
    import multiprocessing
    N = multiprocessing.cpu_count()
    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).imap(_single_compile,objects))
    return objects
import distutils.ccompiler

if PARALLEL:
    distutils.ccompiler.CCompiler.compile = parallelCCompile

# -----------------------------------------------------------------------------
# Global
# -----------------------------------------------------------------------------

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

include_dirs = [
    'loristrck', 
    'src/loristrck',
    'src/loris', 
    numpy_include
]

library_dirs = []
libs = ['m', 'fftw3']
compile_args = ['-DMERSENNE_TWISTER', '-DHAVE_FFTW3_H']

#######################################
# Mac OSX
######################################
if sys.platform == 'darwin':
    include_dirs.extend([
        '/usr/local/include',
        '/opt/local/include'
    ])
    library_dirs.extend([
        '/opt/local/lib'
    ])
######################################
# Windows
######################################
elif sys.platform == 'win32':
    include_dirs.extend([
        '/src/fftw'     # the path of the directory where fftw was unzipped
    ])
    library_dirs.extend([
        '/src/fftw'     # the path of the directory where fftw was unzipped
    ])
    compile_args.append("-march=i686")
    print "NB: make sure that the FFTW dlls are in the windows PATH"

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

loris = Extension(
    'loristrck._core',
    sources=sources + ['loristrck/_core.pyx'],
    include_dirs=include_dirs,
    libraries=libs,
    library_dirs=library_dirs,
    extra_compile_args=compile_args,
    language='c++'
)

doc_lines = __doc__.split('\n')

setup(
    name='loristrck',
    description=doc_lines[0],
    long_description='\n'.join(doc_lines[2:]),
    url='https://github.com/gesellkammer/loristrck',
    download_url='https://github.com/gesellkammer/loristrck',
    license='GPL',
    author='Eduardo Moguillansky',
    author_email='eduardo.moguillansky@gmail.com',
    platforms=['Linux', 'Mac OS-X', 'Unix'],
    version='0.3',
    ext_modules=[loris],
    cmdclass={'build_ext': build_ext},
    packages=['loristrck'],
    install_requires = [ 
        'numpy>=1.6',
        'scipy>=0.10',
        'cython>=0.19'
    ]
)
