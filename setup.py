from setuptools import setup
import os

install_requires = ['pillow', 'numpy']
# if os.environ.get('READTHEDOCS', None) is None:
#     install_requires.append('python-lzo')

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyora',
    author="InkLab",
    author_email="inklabapp@gmail.com",
    version='0.2.1',
    packages=['pyora',],
    license='MIT License',
    description="Read, Write, and Render OpenRaster (.ORA) files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/inklabapp/pyora",
    setup_requires=['wheel'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires
)