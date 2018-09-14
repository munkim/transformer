import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

'''

Modules

'''


def get_position_encoding(embedding_size, d_model):
    radians_matrix = [[pos / 1000 ** (2 * i / d_model) if pos != 0 else 0 for i in range(d_model)] for pos in
                      range(embedding_size)]
    PE_out = [np.sin(pos) if i % 2 == 0 else np.cos(pos) for i, pos in enumerate(radians_matrix)]
    PE_out = np.asarray(PE_out)  # Convert from list to np array
    PE_out = torch.from_numpy(PE_out).type(torch.FloatTensor)  # Convert from np array to Pytorch tensor
    if torch.cuda.is_available():
        PE_out = PE_out.cuda()
    PE_out = Variable(PE_out, volatile=True) # torch Variable
    return PE_out  # dimension = [embedding_size, d_model]



class Attention(nn.Module):
    def __init__(self, d_model, dropout):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.embedding_size = d_model
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, d_k, mask=False):
        # Q, K, V should be 3D tensor.

        _Q = torch.transpose(Q, 0, 2)
        _K = torch.transpose(K, 0, 2)
        _V = torch.transpose(V, 0, 2)

        K_T = torch.transpose(_K, 1, 2)  # K_T = [d_model, batch*max_seq]
        dot = torch.bmm(_Q, K_T)  # [batch*max_seq, batch*max_seq]

        scaled_dot = torch.mul(dot, 1/math.sqrt(d_k))
        if mask == True:
            # Masking --> set the element to negative infinity when the elements in Q or K are negative.
            # "all values in the input of the softmax which corresponds to illegal connections"
            for b in range(len(Q)):
                for r, c in range(self.embedding_size, d_k):
                    if Q[b, r, c] < 0 or K[b, r, c] < 0:
                        #scaled_dot[r, c] = -2 ** (32) + 1
                        scaled_dot[b, r, c] = float('-inf') # Python representation of negative infinity

        _scaled_dot = scaled_dot.view(-1, d_k)
        softmax_out = self.softmax(_scaled_dot)
        softmax_out = softmax_out.view(scaled_dot.size())

        attention_out = torch.bmm(self.dropout(softmax_out), _V)
        attention_out = torch.transpose(attention_out, 0, 2)

        return attention_out


'''

Sublayer Modules

'''


# Ba et al. Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(1, d_model), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, d_model), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z) / self.d_model
        sigma = torch.std(z)
        print (z.size())
        print (mu.size())
        print (sigma.size())
        print (mu.expand_as(z).size())
        layernorm_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        print (layernorm_out.size())
        print (sigma.expand_as(z).size())
        layernorm_out = layernorm_out * self.alpha.expand_as(z) + self.beta.expand_as(z)
        print (self.alpha.size())
        print (layernorm_out.size())
        print (self.alpha.expand_as(z).size())
        exit()
        return layernorm_out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2024):  # Two linear transformations with a ReLU activation in between
        super(FeedForward, self).__init__()
        self.FFN1 = nn.Conv1d(d_model, d_ff, 1)  # Position-wise FeedFWD
        self.ReLU1 = nn.ReLU()
        self.FFN2 = nn.Conv1d(d_ff, d_model, 1)  # Position-wise FeedFWD
        self.ReLU2 = nn.ReLU()

    def forward(self, norm_out):
        FFN1_out = self.FFN1(norm_out.transpose(1,2))
        ReLU1_out = self.ReLU1(FFN1_out)
        FFN2_out = self.FFN2(ReLU1_out).transpose(2,1)
        ReLU2_out = self.ReLU2(max(0, FFN2_out))
        return ReLU2_out


class MultiHead(nn.Module):
    def __init__(self, d_model, H, dropout):  # Depth of Scaled Dot-Product Attention
        super(MultiHead, self).__init__()
        self.H = H
        self.d_q = int(d_model / H)
        self.d_k = int(d_model / H)
        self.d_v = int(d_model / H)

        self.Q_Linears = nn.ModuleList([nn.Linear(d_model, self.d_q) for _ in range(H)])
        self.K_Linears = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(H)])
        self.V_Linears = nn.ModuleList([nn.Linear(d_model, self.d_v) for _ in range(H)])
        self.attentions = nn.ModuleList([Attention(d_model, dropout)for _ in range(H)])

        self.multihead_linear = nn.Linear(self.d_q*self.H, d_model) # Dimension needs to have [d_q * H, d_model]

    def forward(self, Q, K, V, d_model, mask=False):
        Q_original_size, K_original_size, V_original_size = Q.size(), K.size(), V.size()
        Q_return_size = [Q_original_size[0], Q_original_size[1], self.d_q]
        K_return_size = [K_original_size[0], K_original_size[1], self.d_k]
        V_return_size = [V_original_size[0], V_original_size[1], self.d_v]
        Q = Q.view(-1, d_model)
        K = K.view(-1, d_model)
        V = V.view(-1, d_model)
        Q_projected = []
        K_projected = []
        V_projected = []
        head_outputs = []

        for i, (q_linear, k_linear, v_linear, attention) in enumerate(zip(self.Q_Linears, self.K_Linears, self.V_Linears, self.attentions)):
            Q_projected.append(q_linear(Q))
            K_projected.append(k_linear(K))
            V_projected.append(v_linear(V))
            head_outputs.append(attention(Q_projected[i].view(Q_return_size), K_projected[i].view(K_return_size), V_projected[i].view(V_return_size), self.d_k, mask=mask))

        head_concat = torch.cat(head_outputs, 2)
        multihead_out = self.multihead_linear(head_concat.view(-1, d_model))  # the output of concatenate is a linear layer
        multihead_out = multihead_out.view(head_concat.size())
        return multihead_out


'''

Encoder-Decoder

'''


class Encoder(nn.Module):
    def __init__(self, H, d_model, d_ff, dropout):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.multihead = MultiHead(d_model, H, dropout)
        self.layernorm = LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, en_input):
        multihead_out = self.multihead(en_input, en_input, en_input, self.d_model)
        print (multihead_out.size())
        exit()
        multihead_out = self.dropout(multihead_out)
        add_norm_out1 = self.layernorm(torch.add(multihead_out, en_input))
        print (add_norm_out1.size())
        # exit()

        feedforward_out = self.feedforward(add_norm_out1)
        feedforward_out = self.dropout(feedforward_out)
        add_norm_out2 = self.layernorm(torch.add(feedforward_out, add_norm_out1))

        return add_norm_out2


class Decoder(nn.Module):
    def __init__(self, H, d_model, d_ff, dropout):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.multihead = MultiHead(d_model, H, dropout)
        self.layernorm = LayerNorm(d_model, d_ff)
        self.feedforward = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, de_input, en_output):
        # src_original_size = en_output.size()
        # en_output = en_output.view(-1, self.d_model) # Resize to 2D for Linear Transformation
        #
        # tgt_original_size = de_input.size()
        # de_input = de_input.view(-1, self.d_model) # Resize to 2D for Linear Transformation

        # All outputs have the dimension of 512 (d_model).
        multihead_out1 = self.multihead(de_input, de_input, de_input, self.d_model, mask=True)
        multihead_out1 = self.dropout(multihead_out1)
        # print("\n\n################ MULTIHEAD OUT 1 ###############")
        # print(multihead_out1)
        add_norm_out1 = self.layernorm(torch.add(multihead_out1, de_input))
        # print("\n###### DECODER NORM OUT 1")
        # print(add_norm_out1)

        multihead_out2 = self.multihead(add_norm_out1, en_output, en_output, self.d_model)
        multihead_out2 = self.dropout(multihead_out2)
        # print("\n\n\n\n################ MULTIHEAD OUT 2 ###############")
        # print(multihead_out2)
        add_norm_out2 = self.layernorm(torch.add(multihead_out2, add_norm_out1))
        # print("\n###### DECODER NORM OUT 2")
        # print(add_norm_out2)
        # add_norm_out2 = add_norm_out2.view(tgt_original_size) # Need to convert back to 3D for position-wise feedfwd

        feedforward_out = self.feedforward(add_norm_out2)
        feedforward_out = self.dropout(feedforward_out)
        add_norm_out3 = self.layernorm(torch.add(feedforward_out, add_norm_out2))
        # print("\n###### DECODER NORM OUT 3")
        # print(add_norm_out3)

        return add_norm_out3


'''

Transformer

'''


class Transformer(nn.Module):
    def __init__(self, transformer_config):
        super(Transformer, self).__init__()
        '''
        ### transformer_config ###
        'N': 6,
        'd_model': 512,
        'd_ff': 2024,
        'H': 8,
        'batch_size': batch_size,
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab),
        'max_seq': 100,
        'dropout': 0.1,
        'use_cuda': True
        '''
        # for python3
        # self.N, self.d_model, self.d_ff, self.H, self.batch_size, self.src_vocab_size, self.tgt_vocab_size, self.max_seq, self.dropout, self.use_cuda = transformer_config.values()

        # for python2
        self.N, self.d_model, self.d_ff, self.H, self.batch_size, self.src_vocab_size, self.tgt_vocab_size, self.max_seq, self.dropout, self.use_cuda = transformer_config
        self.src_embedding_size = self.src_vocab_size
        # print("\n\nSRC EMBEDDING SIZE (VOCAB SIZE)")
        # print(self.src_embedding_size)
        self.src_embedding = nn.Embedding(self.src_embedding_size, self.d_model)
        self.tgt_embedding_size = self.tgt_vocab_size
        self.tgt_embedding = nn.Embedding(self.tgt_embedding_size, self.d_model)

        # Use pretrained model (Gensim or SpaCy required)
        # self.embedding.weights.data.copy_(torch.from_numpy(en_model)) # Copy the learned weights and use it for the embedding.
        # self.embedding.weight.requires_grad = False
        # parameters = filter(lambda p: p.requires_grad, net.parameters()) # Freeze the embedding matrix during training.

        self.encoder_stacks = nn.ModuleList(
            [Encoder(self.H, self.d_model, self.d_ff, self.dropout) for _ in range(self.N)]
        )
        self.decoder_stacks = nn.ModuleList(
            [Decoder(self.H, self.d_model, self.d_ff, self.dropout) for _ in range(self.N)]
        )

        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(self.d_model, self.tgt_vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, input_batch, target_batch):
        # print("\n\nINPUT BATCH")
        # print(input_batch)
        # # exit()
        # Define inputs to the first Encoder stack
        encoder_inputs = self.src_embedding(input_batch.long())  # Embedding output size [batch_size, max_seq, d_model]
        # For every batch of the source data, add position encoding
        for j in range(len(encoder_inputs)):
            pos_enc = get_position_encoding(len(encoder_inputs[j]), self.d_model)
            encoder_inputs[j] = encoder_inputs[j] + pos_enc
        encoder_inputs = self.dropout(encoder_inputs)

        # Define inputs to the first Decoder stack (Shifted Output)
        decoder_inputs = self.tgt_embedding(target_batch.long())  # Embedding output size [batch_size, max_seq, d_model]
        # For every batch of the source data, add position encoding
        for k in range(len(decoder_inputs)):
            pos_enc = get_position_encoding(len(decoder_inputs[j]), self.d_model)
            decoder_inputs[k] = decoder_inputs[k] + pos_enc
        decoder_inputs = self.dropout(decoder_inputs)

        # For the number of stacks
        encoder_outputs = []
        decoder_outputs = []
        for i, (encoder, decoder) in enumerate(zip(self.encoder_stacks, self.decoder_stacks)):
            if i == 0:
                # print("\n\nDECODER %d" % i)
                # print("\n## DECODER INPUTS (%d)" % i)
                # print(decoder_inputs)
                encoder_outputs.append(encoder(encoder_inputs))
                decoder_outputs.append(decoder(decoder_inputs, encoder_outputs[i]))
            else:
                # print("\n\nDECODER %d" % i)
                # print("\n\n## DECODER INPUTS (%d)" % i)
                # print(decoder_outputs[i - 1])
                encoder_outputs.append(encoder(encoder_outputs[i - 1]))
                decoder_outputs.append(decoder(decoder_outputs[i - 1], encoder_outputs[i]))
                # print("\n#### DECODER OUTPUTS (%d)" % i)
                # print(decoder_outputs[i])


        last_decoder_output = decoder_outputs[i-1].view(-1, self.d_model)  # Resize to 2D for Linear Transformation
        output_linear = self.linear(last_decoder_output)

        return output_linear
