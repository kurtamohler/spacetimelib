from setuptools import setup, find_packages

from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}


with open('README.md') as f:
    readme = f.read()

# TODO: Decide which license to use
# with open('LICENSE') as f:
#    license = f.read()

name = 'spacetime'
version = '0.0'
release = '0.0.0'

setup(
    name=name,
    version=version,
    description='Special relativity compute library',
    long_description=readme,
    author='Kurt Mohler',
    author_email='kurtamohler@gmail.com',
    url='https://github.com/kurtamohler/spacetimelib',
    # license=license,
    packages=['spacetime'],
    cmdclass=cmdclass,
    # these are optional and override conf.py settings
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'doc/source')}},
)

