from setuptools import setup
import shutil


__project__ = "HPS_writer"
__version__ = "1.0.0"
__description__ = "Testing"
__packages__ = ["md"]
__required__ = ["mdtraj","pathos","mdanalysis"]

setup(
    name = __project__,
    version = __version__,
    description = __description__,
    packages = __packages__,
    install_requires = __required__,
)

shutil.rmtree("build")
shutil.rmtree("dist")
shutil.rmtree("HPS_writer.egg-info")