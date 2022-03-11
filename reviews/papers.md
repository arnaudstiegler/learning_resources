# Perceiver

The perceiver is a new transformer-based architecture which can handle any modality: 
any input sequence L*D (L sequence length, D dimension) will be mapped into a latent space K*D (K fixed value, usually K<<L), and then followed by the usual self-attention.
**By alternating cross-attention and self-attention, the model has a linear complexity wrt sequence length**, but retains enough expressivity through repeated querying
the original input sequence.
For each input type (text, image, etc..), you need a preprocessing function that will turn it into a sequence of tokens. So even if the model/architecture is 
agnostic to the input-type, there is still a need for an input-specific preprocessor.

# Sentence Bert:

Bert can't handle semantic similarity tasks or clustering texts because it would require too many inference paths (one for each sentence pair). To make it tractable, SentenceBert is trained so that the pooled sentence embedding is meaningful and can be directly used for classification: two similar sentences will have similar sentence embeddings, and therefore you just need to run inference once per sentence instead of one per pair of sentence. The SBert model is a Bert model fine-tuned on SNLI using a siamese network that encodes both sentences and use (u, v, u-v) as vector to classify the pair (with normal cross-entropy loss). It can also be used with MeanSquared error or Triplet loss. Tests on SentEval benchmarks have shown that sentence embeddings were much more meaningful than the CLS embedding from Bert.

# DeepNet: Scaling Transformers to 1,000 layers

Very large Transformer-based model are still relatively shallow as adding more layers lead to instability. The paper shows that it's a combination of having very large model updates at early stage (proportional to the number of layers), followed by vanishing gradients which prevents the model from exiting the suboptimal local minima.
To remedy that, the paper shows that we can theoretically bound the norm of the model updates for every layer with a constant, using:
- the correct initialization for the different weights
- scaling the residual connection inside the LayerNorm with a scaling factor alpha
With those updates, the model updates remain bounded throughout all layers of the model, avoiding instabilities mentioned before.
