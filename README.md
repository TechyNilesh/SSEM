# SSEM (Semantic Similarity Based Evaluation Metrics)

![Generic badge](https://img.shields.io/badge/HuggingFace-NLP-yellow.svg) ![Generic badge](https://img.shields.io/badge/Python-V3.10-blue.svg) ![Generic badge](https://img.shields.io/badge/pip-V3-red.svg)  ![Generic badge](https://img.shields.io/badge/Transformers-V4-orange.svg) ![Generic badge](https://img.shields.io/badge/Gensim-V4-blueviolet.svg) [![Downloads](https://static.pepy.tech/personalized-badge/ssem?period=total&units=none&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/ssem)

SSEM is a python library that provides evaluation metrics for natural language processing (NLP) text generation tasks with support of multiple languages. The library focuses on measuring the semantic similarity between generated text and reference text. It supports various distance metrics, such as cosine similarity, euclidean distances, and pearson correlation.

The library is built on top of the popular Hugging Face Transformers library and is compatible with any pre-trained transformer model. Additionally, it supports parallel processing for faster computation and offers multiple evaluation levels, such as sentence-level, token-level, and Latent Semantic Indexing (LSI) based similarity.

## Developed By

### [Nilesh Verma](https://nileshverma.com "Nilesh Verma")

## Features

- Compatible with any Hugging Face pre-trained transformer models.
- Multiple language support.
- Supports multiple distance metrics: cosine, euclidean, and Pearson correlation.
- Supports different levels of evaluation: sentence-level, token-level, and LSI (Latent Semantic Indexing).
- Supports parallel processing for faster computation.
- Customizable model embeddings.

## Installation

You can install the SSEM library using pip:

```
pip install ssem
```

## How to use SSEM

To use SSEM, you first need to import the library and create an instance of the `SemanticSimilarity` class. You can specify the pre-trained model you want to use, the distance metric, and any custom embeddings.

```python
from ssem import SemanticSimilarity

ssem = SemanticSimilarity(model_name='bert-base-multilingual-cased', metric='cosine',custom_embeddings=None)
```

Once you have created an instance, you can use the `evaluate()` method to calculate the similarity between the list of generated text and list of reference text. You can specify various options such as the number of parallel jobs, the evaluation level, and the output format.

```python
output_sentences = ['This is a generated sentence 1.','This is a generated sentence 2.']
reference_sentences = ['This is the reference sentence 1.','This is the reference sentence 2.']

similarity_score = ssem.evaluate(output_sentences, reference_sentences, n_jobs=1, level='sentence', output_format='mean')
```

The `evaluate()` method returns a similarity score, which can be a single float value (mean), a standard deviation value (std), or both (mean_std). 

```python
print("Similarity score: ", similarity_score)
```

You can use this score to assess the quality of the generated text compared to the reference text.

### Parameters

- `model_name`: The name of the pre-trained transformer model to use. Default is `'bert-base-multilingual-cased'`.
- `metric`: The similarity metric to use. Options are `'cosine'`, `'euclidean'`, and `'pearson'`. Default is `'cosine'`.
- `custom_embeddings`: An optional numpy array containing custom embeddings. Default is `None`.
- `n_jobs`: The number of parallel jobs to use for processing. Default is `1`.
- `level`: The level of evaluation to perform. Options are `'sentence'`, `'token'`, and `'lsi'`. Default is `'sentence'`.
- `output_format`: The format of the output. Options are `'mean'`, `'std'`, and `'mean_std'`. Default is `'mean'`.

## License

SSEM is released under the MIT License.


## References

1. [Evaluation Measures for Text Summarization](https://www.researchgate.net/publication/220106310_Evaluation_Measures_for_Text_Summarization)
2. [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
3. [Semantic Similarity Based Evaluation for Abstractive News Summarization](https://aclanthology.org/2021.gem-1.3/)
4. [Evaluation of Semantic Answer Similarity Metrics](https://arxiv.org/abs/2206.12664)

### Please do STAR the repository, if it helped you in anyway.

More cool features will be added in future. Feel free to give suggestions, report bugs and contribute.