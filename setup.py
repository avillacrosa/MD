from setuptools import setup
import subprocess
import sys
import shutil
import os



subprocess.check_call([sys.executable, "-m", "pip", "install", "mdtraj"])

setup(
    name = "HPS_writer",
    version = "1.0.0",
    description = "IQAC md writer",
    packages = ["md"],
    install_requires = ["pathos","mdanalysis"],
    package_data={'': [x[0].replace('md/','')+"/*" for x in os.walk('md/data')]}
)


shutil.rmtree('build')
shutil.rmtree('dist')
shutil.rmtree('HPS_writer.egg-info')
