#setup.py
from setuptools import setup, find_packages
setup(
    name='lilab', # 应用名
    version='0.2', # 版本号
    packages=find_packages(include=['LILAB*']), # 包括在安装包内的 Python 包
    author='chenxf',
    author_email='cxf529125853@163.com',
    url='https://pypi.org/project/lilab/'
)
