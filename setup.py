from setuptools import setup, find_packages

setup(
    name="xml_augmenter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==2.0.0",
        "opencv_python==4.10.0.84",
    ],
    author="nastics",
    description="Simple XML file augmentation handler for segmentation based CV tasks",
    long_description=open("README.md").read(),
    long_description_content_type= "text/markdown"
)
