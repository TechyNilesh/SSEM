import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import jaccard_score, pairwise_distances
from scipy.spatial.distance import correlation
from multiprocessing import Pool
from typing import List, Union, Callable
from gensim import corpora, models, similarities as sm
from tqdm import tqdm

class SemanticSimilarity:
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', metric: str = 'cosine', custom_embeddings: np.ndarray = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.metric = metric
        self.custom_embeddings = custom_embeddings

    def _encode(self, sentences: List[str]) -> np.ndarray:
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return embeddings

    def _similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        if self.metric == 'cosine':
            return cosine_similarity(embeddings1, embeddings2)
        elif self.metric == 'euclidean':
            return euclidean_distances(embeddings1, embeddings2)
        elif self.metric == 'pearson':
            return np.array([[correlation(x, y) for y in embeddings2] for x in embeddings1])
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

    def _process_chunk(self, chunk: List[str]) -> np.ndarray:
        if self.custom_embeddings is None:
            embeddings = self._encode(chunk)
        else:
            embeddings = self.custom_embeddings
        return embeddings

    def _parallel_processing(self, sentences: List[str], n_jobs: int) -> np.ndarray:
        with Pool(n_jobs) as p:
            embeddings = p.map(self._process_chunk, sentences)
        return np.vstack(embeddings)

    def _calculate_mean_std(self, similarities: np.ndarray) -> float:
        mean = np.mean(similarities)
        std = np.std(similarities)
        return mean, std

    def evaluate(self, output_sentences: List[str], reference_sentences: List[str], n_jobs: int = 1, level: str = 'sentence', output_format: str = 'mean') -> Union[float, np.ndarray]:
        if level == 'sentence':
            if n_jobs > 1:
                output_embeddings = self._parallel_processing(output_sentences, n_jobs)
                reference_embeddings = self._parallel_processing(reference_sentences, n_jobs)
            else:
                output_embeddings = self._process_chunk(output_sentences)
                reference_embeddings = self._process_chunk(reference_sentences)
            similarities = self._similarity(output_embeddings, reference_embeddings)
        elif level == 'token':
            token_similarities = []
            for output, reference in tqdm(zip(output_sentences, reference_sentences)):
                output_tokens = self.tokenizer.tokenize(output)
                reference_tokens = self.tokenizer.tokenize(reference)
                token_similarities.append(self.evaluate(output_tokens, reference_tokens, n_jobs, level='sentence'))
            similarities = token_similarities
        elif level == 'lsi':
            lsi_similarities = []
            tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in output_sentences + reference_sentences]
            dictionary = corpora.Dictionary(tokenized_sentences)
            corpus = [dictionary.doc2bow(sentence) for sentence in tokenized_sentences]
            lsi = models.LsiModel(corpus, id2word=dictionary)
            index = sm.MatrixSimilarity(lsi[corpus[:len(output_sentences)]])
            for reference in tqdm(reference_sentences):
                vec_bow = dictionary.doc2bow(self.tokenizer.tokenize(reference))
                vec_lsi = lsi[vec_bow]
                sims = index[vec_lsi]
                lsi_similarities.append(sims)
            similarities = lsi_similarities
        else:
            raise ValueError(f"Invalid level: {level}")

        if output_format == 'mean':
            return np.mean(similarities)
        elif output_format == 'std':
            return np.std(similarities)
        elif output_format == 'mean_std':
            return self._calculate_mean_std(similarities)
        else:
            raise ValueError(f"Invalid output_format: {output_format}")