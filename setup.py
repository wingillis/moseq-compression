from setuptools import setup


setup(
    name='moseq-compress',
    version='0.1.2',
    author='Winthrop Gillis',
    py_modules=['main'],
    install_requires=[
        'click',
        'h5py',
        'numpy',
        'av',
        'tqdm',
        'opencv-python==4.1.2.30',
    ],
    entry_points='''
        [console_scripts]
        moseq-compress=main:main
    ''',
    url='https://github.com/wingillis/moseq-compression'
)