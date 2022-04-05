from setuptools import setup

install_requires = [
        "numpy",
        "scipy>=1.3",
        "numba>=0.49",
        "torch",
        "torch-scatter",
        "torch-sparse",
        "scikit-learn",
        "sacred",
        "seml"
]

setup(
        name='pprgo_pytorch',
        version='1.0',
        description='PPRGo model in PyTorch, from "Scaling Graph Neural Networks with Approximate PageRank"',
        author='Aleksandar Bojchevski, Johannes Gasteiger, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann',
        author_email='a.bojchevski@in.tum.de, j.gasteiger@in.tum.de',
        packages=['pprgo'],
        install_requires=install_requires,
        zip_safe=False
)
