# Perceiver

The perceiver is a new transformer-based architecture which can handle any modality: 
any input sequence L*D (L sequence length, D dimension) will be mapped into a latent space K*D (K fixed value, usually K<<L), and then followed by the usual self-attention.
**By alternating cross-attention and self-attention, the model has a linear complexity wrt sequence length**, but retains enough expressivity through repeated querying
the original input sequence.
For each input type (text, image, etc..), you need a preprocessing function that will turn it into a sequence of tokens. So even if the model/architecture is 
agnostic to the input-type, there is still a need for an input-specific preprocessor.
