from setuptools import setup
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
  long_description_content_type="text/markdown",
  name = 'SSEM',         
  packages = ['SSEM'],
  version = '1.0',
  license='MIT',        
  description = 'SSEM is a semantic similarity-based evaluation library for natural language processing (NLP) text generation tasks. It supports various similarity metrics and evaluation levels, and is compatible with any Hugging Face pre-trained transformer model.',
  long_description=README,
  author = 'Nilesh Verma',                   
  author_email = 'me@nileshverma.com',     
  url = 'https://github.com/TechyNilesh/SSEM',
  download_url = 'https://github.com/TechyNilesh/SSEM/archive/refs/tags/v_1.tar.gz',    
  keywords = ['Semantic similarity', 'SSEM', 'Evaluation metrics','NLP'],   
  install_requires=[        
          'gensim',
          'numpy',
          'scikit_learn',
          'scipy',
          'transformers',
          'tqdm',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)