from setuptools import setup

setup(name='py_openai_extractor',
    version='0.0.5',
    description='news extractor using openai',
    author='PortADa team',
    author_email='jcbportada@gmail.com',
    license='MIT',
    url="https://github.com/portada-git/py_portada_openai_extractor.git",
    packages=['py_openai_extractor'],
    py_modules=['extractor', 'portada_autonewsextractor_adaptor'],
    install_requires=[
	    'openai',
        'babel',
    ],
    python_requires='>=3.9',
    zip_safe=False)
