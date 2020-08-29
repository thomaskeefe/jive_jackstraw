from setuptools import setup

setup(name='jive_jackstraw',
      version='0.1',
      description='Jackstraw methods for JIVE',
      url='http://github.com/thomaskeefe/jive_jackstraw',
      author='Thomas Keefe',
      author_email='tkeefe@live.unc.edu',
      packages=['jive_jackstraw'],
      install_requires=[
        'numpy',
        'numba',
        'tqdm'
      ])
