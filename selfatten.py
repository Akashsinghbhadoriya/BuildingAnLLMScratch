import torch
import torch.nn as nn


#Compact Self attention class
class SelfAttention_v1(nn.Module) :
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in,d_out))
        self.W_key = nn.Parameter(torch.rand(d_in,d_out))
        self.W_value = nn.Parameter(torch.rand(d_in,d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)

        #masking values above diagonal for causal attention masking using zeros
        context_length = keys.shape[0]
        mask_simple = torch.tril(torch.ones(context_length,context_length))
        
        masked_simple = attn_weights * mask_simple
        
        row_sums = masked_simple.sum(dim=-1, keepdim=True)
        attn_weight = masked_simple / row_sums
        

        context_vect = attn_weights @ values
        return context_vect
    
#Note that nn.Linear in SelfAttention_v2 uses a different weight initialization scheme as nn.Parameter(torch.rand(d_in,d_out)) used in SelfAttention_v1, which causes both mechanisms to produce different results.
class SelfAttention_v2(nn.Module) :
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T

        #masking values above diagonal for causal attention masking using -inf
        context_length = keys.shape[0]
        mask = torch.triu(torch.ones(context_length,context_length), diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        
        attn_weights = torch.softmax(masked/keys.shape[-1]**0.5, dim=-1)
       
        
        context_vect = attn_weights @ values
        return context_vect
    
# compact casual attention class added batching
class CasualAttention(nn.Module) :
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length,context_length), diagonal=1)
        )

    def forward(self, x) :
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool() [:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vect = attn_weights @ values

        return context_vect
    
#multihead attention wrapper
class MultiHeadAttentionWrapper(nn.Module) :
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias =False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CasualAttention(d_in,d_out,context_length,dropout,qkv_bias
                            ) for _ in range(num_heads)]
        )

    def forward(self,x) :
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

# Efficient implementation of Multihead attention by computing the matrix multiplication in one step instead of doing it separately for different heads
class MultiheadAttention(nn.Module) :
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False) :
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length), diagonal=1)
        )

    def forward(self, x) :
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #changes the dimension of the keys array for multiplication of heads in one go
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2)  #keys.view(b, self.num_heads, num_tokens, self.head_dim)
        queries = queries.transpose(1,2) #queries.view(b, self.num_heads, num_tokens, self.head_dim)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)  #queries.view(b, self.num_heads, num_tokens, self.head_dim) @ keys.view(b, self.num_heads, self.head_dim, num_tokens)
        # result of above matrix multiplication is attn_scores(b, self.num_heads, num_tokens, num_tokens)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5 , dim=-1)

        attn_weights = self.dropout(attn_weights)
        context_vect = (attn_weights @ values ).transpose(1,2) # changing the dimensions to the original format
        context_vect = context_vect.contiguous().view(b,num_tokens, self.d_out) # making the context vector continuous

        context_vect = self.out_proj(context_vect) #blends the independent insights from individual heads into single unified representation
        return context_vect
    

if __name__ == "__main__" :
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]] # step (x^6)
    )

    #intermediate attention weights for query 2 for all the inputs using a dot product
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, i_n in enumerate(inputs) :
        attn_scores_2[i] = torch.dot(i_n, query)

    # print(attn_scores_2)
    attn_weights_2_temp = attn_scores_2 / attn_scores_2.sum()
    # print(attn_weights_2_temp)

    #intermediate attention weight normalization using softmax function for normalization
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

    # print(attn_weights_2)

    #calculating the context vector 2
    context_vect_2 = torch.zeros(query.shape)
    for i , x_i in enumerate(inputs) :
        context_vect_2 += attn_weights_2[i] * x_i

    # print(context_vect_2)

    # calculating the attention weights and context vectors for all the inputs

    attn_scores = torch.empty(inputs.shape[0],inputs.shape[0])
    # print(attn_scores.shape)

    # for i in range(inputs.shape[0]) :
    #     for j, x_j in enumerate(inputs) :
    #         attn_scores[i][j] = torch.dot(x_j, inputs[i])

    attn_scores = inputs @ inputs.T

    # print(attn_scores)

    attn_weights = torch.softmax(attn_scores, dim=-1)

    # print(attn_weights)

    context_vect = torch.zeros(inputs.shape)

    context_vect = attn_weights @ inputs

    # print(context_vect)

    #attention mechanism using trainable weights

    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False) #requires_grad=True for update matrices during model training

    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value

    # print(query_2)

    Keys = inputs @ W_key
    Values = inputs @ W_value 

    attn_score_22 = query_2.dot(key_2)

    # print(attn_score_22)

    #here the attention score is calculated by the matrix multiplication of query and keys generated using weight matrices
    attn_score_2 = query_2 @ Keys.T

    # print(attn_score_2)

    d_k = Keys.shape[-1]
    #attn weight is normalized using the square root of the dimensions of the embedding
    attn_weight_2 = torch.softmax(attn_score_2/d_k**0.5, dim=0)

    # print(attn_weight_2)

    #context vector is calculated by the weighted sum over the value vectors
    context_vect2 = attn_weights_2 @ Values

    # print(context_vect2)

    torch.manual_seed(123)
    self_aatn = SelfAttention_v1(d_in,d_out)
    # print(self_aatn(inputs).shape)

    torch.manual_seed(789)
    self_attnV2 = SelfAttention_v2(d_in,d_out)
    # print(self_attnV2(inputs))

    #dropout example
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5) #1
    example = torch.ones(6, 6) #2
    # print(dropout(example))

    batch = torch.stack((inputs,inputs),dim=0)
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CasualAttention(d_in, d_out, context_length, 0.0)
    context_vect = ca(batch)
    # print(context_vect)

    torch.manual_seed(123)
    mha = MultiHeadAttentionWrapper(d_in,d_out,context_length,0.0,2)
    context_vects = mha(batch)
    # print(context_vects.shape)
    # print(context_vects)

    torch.manual_seed(123)
    bs, cl, din = batch.shape
    dout = 2
    multi_head = MultiheadAttention(din,dout,cl,0.0,num_heads=2)
    c_v = multi_head(batch)
    # print(c_v)