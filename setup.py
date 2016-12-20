
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Cerebro',
    'author': 'Nhan Vu',
    'url': 'http://github.com/nhanitvn/cerebro',
    'download_url': 'http://github.com/nhanitvn/cerebro',
    'author_email': 'nhanitvn@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['cerebro'],
    'scripts': [],
    'name': 'projectname'
}

setup(**config, requires=['numpy'])
