
from setuptools import setup, find_packages

setup(
    name='crspy_lite',
    version='0.1.4',
    description='A single site Cosmic-Ray Neutron Sensing (CRNS) processing tool for soil moisture estimation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Joe Wagstaff',
    author_email='joe.d.wagstaff@gmail.com',
    url='https://github.com/Joe-Wagstaff/crspy-lite',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'bokeh',
        'beautifulsoup4',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'crspy_lite=crspy_lite_code.full_process_wrapper:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
