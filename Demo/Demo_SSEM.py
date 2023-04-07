# import SSEM
from SSEM import SemanticSimilarity

# Create an instance of SemanticSimilarity
similarity = SemanticSimilarity(model_name='bert-base-multilingual-cased', metric='cosine',custom_embeddings=None)

# Define the sentences to compare
output_sentences = ["This is a sentence.", "Another sentence here."]
reference_sentences = ["This sentence is similar.", "A different sentence."]

# Compute the similarity score
result = similarity.evaluate(output_sentences, reference_sentences, level='sentence', output_format='mean',n_jobs=1)
# Print the result
print("Mean Sentence Similarity:", result)

# Compute the similarity score
result = similarity.evaluate(output_sentences, reference_sentences, level='token', output_format='mean',n_jobs=1)
# Print the result
print("Mean Token Similarity:", result)

# Compute the similarity score
result = similarity.evaluate(output_sentences, reference_sentences, level='lsi', output_format='mean',n_jobs=1)
# Print the result
print("Mean LSI Similarity:", result)


# In[ ]:




