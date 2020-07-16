from setuptools import setup

# Since CUDA 10.0 includes a bug that affects PPRGo we strongly recommend using e.g. 10.1,
# which can be installed using environment.yaml (e.g. via `conda env create -f environment.yaml`).
install_requires = [
        "numpy",
        "scipy>=1.3",
        "numba>=0.49",
        "pytorch",
        "pytorch_sparse",
        "scikit-learn",
        "sacred",
        "seml"
]

setup(
        name='pprgo_pytorch',
        version='1.0',
        description='PPRGo model in PyTorch, from "Scaling Graph Neural Networks with Approximate PageRank"',
        author='Aleksandar Bojchevski, Johannes Klicpera, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann',
        author_email='a.bojchevski@in.tum.de, klicpera@in.tum.de',
        packages=['pprgo'],
        install_requires=install_requires,
        zip_safe=False
)
