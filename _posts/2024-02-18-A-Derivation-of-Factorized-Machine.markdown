---
layout: post
title:  "A Derivation and Implementation of FM"
date:   2024-02-18 20:53:00 +0800
katex: true
---

To model interactions of high-cardinality categorical features (often called as fields), the factorization machine formula can be expressed as 

$y = w_0 + \sum_{i=1}^F \mathbf{W_i} \boldsymbol{x_i} + \sum_{i=1}^F\sum_{j<i}\boldsymbol{x_i}\mathbf{V_i}^T\mathbf{V_j}\ \boldsymbol{x_j}$

where $\boldsymbol{x_i}$ and $\boldsymbol{x_j}$ are one-hot vectors, $\mathbf{W}_i$ is of shape $(\text{vocab}_i \times 1)$, $\mathbf{V}_i$ is of shape $(\text{vocab}_i \times K)$, $K$ is the size of the embedding space, $F$ is the total number of fields.

$\mathbf{W_i}$ and $\mathbf{V_i}$ can be thought as look up tables, and to make it even simpler using parameter sharing among fields, 

$y = w_0 + \sum_{i=1}^F \mathbf{W} \boldsymbol{x_i} + \sum_{i=1}^F \sum_{j < i} \boldsymbol{x_i} \mathbf{V}^T\mathbf{V} \boldsymbol{x_j}$

The feature cross term can be express as $\sum_{i=1}^F\sum_{j<i}(\mathbf{V} \boldsymbol{x_i})^T(\mathbf{V} \boldsymbol{x_j})$.

Let $\boldsymbol{v_i} = \mathbf{V} \boldsymbol{x_i}$ (the embedding vector of $\boldsymbol{x_i}$), then the cross term is

$\sum_{i=1}^F\sum_{j<i}\boldsymbol{v_i}^T \boldsymbol{v_j} = \sum_{i=1}^F \sum_{j < i} \sum_{k=1}^K v_{ik}v_{jk}$

Notice that $\sum_{j<i} (\cdot) = \dfrac{1}{2}[\sum_{\forall j} (\cdot)- \sum_{i = j} (\cdot)]$, then the cross term is

$\dfrac{1}{2}\sum_{k=1}^K [(\sum_{i=1}^F v_{ik})^2 - \sum_{i=1}^F v_{ik}^2]$

So here we have two kinds of summation: $\sum_{k=1}^K$ is summation over embedding axes and $\sum_{i=1}^F$ is summation over fields.

Notice that, in popular implementations, the different fileds are often embedded in the sampe embedding space (parameter sharing or wegiht sharing). 
```python
import torch
import torch.nn as nn

BATCH_AXIS = 0
FIELD_AXIS = 1
EMBED_AXIS = 2
    
class FM(nn.Module):
    
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.vocab_size = sum(field_dims)
        
        self.linear_lookup = nn.Embedding(self.vocab_size, 1)
        self.cross_loopup = nn.Embedding(self.vocab_size, embed_dim)
        self.bias = nn.Parameter(torch.empty(1, 1))

    def forward(self, x):
        linear_term = self.linear_lookup(x).sum(FIELD_AXIS)
        embedding = self.cross_loopup(x) # SHAPE: (B, F, E)
        
        outer_square = embedding.sum(FIELD_AXIS, keepdim=True) ** 2  
        inner_square = (embedding ** 2).sum(FIELD_AXIS, keepdim=True)
        cross_term = 0.5 * (outer_square - inner_square).sum(EMBED_AXIS)
        
        return self.bias + linear_term + cross_term
```

