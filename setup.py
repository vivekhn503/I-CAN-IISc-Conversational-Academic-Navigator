# setup.py
from setuptools import setup, find_packages

setup(
    name="ican_navigator",              
    version="0.1.0",
    package_dir={"": "src"},            
    packages=find_packages(where="src"),
    install_requires=[
        # copy  runtime deps from requirements.txt, e.g.:
        "streamlit",
        "langchain",
        "python-dotenv",
        "langchain-community"
    ],
)
