from setuptools import setup, find_packages

setup(name='AquabyteBiomass',
      version='0.1',
      description='Length and biomass estimation',
      author='Thomas Hossler',
      author_email='thomas@aquabyte.ai',
      url='https://github.com/aquabyteai/aquabyte_biomass',
      packages=find_packages(),
      install_requires=['numpy==1.14.0', 'opencv-python==3.4.0.12']
     )