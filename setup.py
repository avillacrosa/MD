from setuptools import setup
import subprocess
import sys


__project__ = "HPS_writer"
__version__ = "1.0.0"
__description__ = "IQAC md writer"
__packages__ = ["md"]
__required__ = ["mdanalysis","mdtraj"]

subprocess.check_call([sys.executable, "-m", "pip", "install", "mdtraj"])

setup(
    name = __project__,
    version = __version__,
    description = __description__,
    packages = __packages__,
    install_requires = ["pathos","mdanalysis"],
)
