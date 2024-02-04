from setuptools import setup
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess


# Custom command to run after package installation
class PostInstallCommand(install):
    def run(self):
        install.run(self)
        self._post_install()

    def _post_install(self):
        print("Running post-install steps...")
        self._install_dependencies()
        self._init_git_submodules()

    def _install_dependencies(self):
        print("Installing necessary packages...")
        subprocess.call(["pip", "install", "numpy", "requests", "poetry"])  # Add your required packages here
        subprocess.call(["pip", "install", "-r","requirements.txt"])  # Add your required packages here

    def _init_git_submodules(self):
        print("Initializing Git submodules...")
        subprocess.call(["git", "submodule", "update", "--init", "--recursive"])


setup(
    name='dermsynth3d',
    version='0.0.3',
    description='DermSynth3D: A 3D Synthetic Dermoscopy Image Dataset',
    url='https://github.com/sfu-mial/DermSynth3D',
    author='Ashish Sinha',
    author_email='ashishsinha108@gmail.com',
    license='Apache 2.0',
    packages=['dermsynth3d'],
    install_requires=["setuptools>=41.0.0"],
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    cmdclass={
        'install': PostInstallCommand,
        # 'develop': PostDevelopCommand,
    },
)
