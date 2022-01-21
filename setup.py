import os
import re
import sys
import platform
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        # The format is (long option, short option, description).
        ('install-prefix=', None, 'Path where C++ library will be installed'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        build_ext.initialize_options(self)
        # Each user option must be listed here with their default value.
        self.install_prefix = None

    def finalize_options(self):
        '''Overloaded build_ext implementation to append custom openssl
        include file and library linking options'''

        build_ext.finalize_options(self)
        if self.install_prefix:
            assert os.path.exists(self.install_prefix), (
                'Install prefix path %s does not exist.' % self.install_prefix)

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = \
            os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DTESTS_ENABLED=OFF']

        avx2_enabled = os.environ.get("AVX2_ENABLED")
        if avx2_enabled is not None:
            cmake_args.append(f"-DAVX2_ENABLED={avx2_enabled}")

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            if os.environ.get('CMAKE_TOOLCHAIN_FILE') is not None:
                cmake_toolchain_file = os.environ.get('CMAKE_TOOLCHAIN_FILE')
                # print(f'-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file}')
                cmake_args += \
                    [f'-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file}']
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                                                        cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                if os.environ.get('CMAKE_TOOLCHAIN_FILE') is not None:
                    cmake_args += ['-DVCPKG_TARGET_TRIPLET=x64-windows']
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        if self.install_prefix:
            subprocess.check_call(["sudo", 'cmake', '--install', '.',
                                   "--prefix", self.install_prefix],
                                  cwd=self.build_temp)


root = Path(__file__).parent
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()
with open(str(root / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

setup(
    name='pixsfm',
    version='0.0.1',
    author='Philipp Lindenberger, Paul-Edouard Sarlin',
    author_email='phil.lindenberger@gmail.com',
    description='Pixel-Perfect Structure-from-Motion',
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["pixsfm", "pixsfm.*"]),
    python_requires='>=3.6',
    ext_modules=[CMakeExtension('pixsfm._pixsfm')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    include_package_data=True,
    package_data={'': ['configs/*.yaml']},
)
