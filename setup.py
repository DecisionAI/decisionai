import setuptools
import decisionai

setuptools.setup(
        name="decisionai",
        url="https://github.com/decisionai/decisionai",
        packages=setuptools.find_packages(),
        version=decisionai.__version__,
        python_requires='>=3.7',
)
