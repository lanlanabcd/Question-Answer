import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

"""Init: batch_size, input_size, hidden_size, bidirectional=True, batch_first=True"""
"""Forward: vec_seq, len_seq"""


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, batch_first=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                            batch_first=batch_first)
        # self.hidden = self.init_hidden()
        #
        # self.batch_size = batch_size
        # self.layer_num = 2 if bidirectional else 1

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.layer_num, self.batch_size, self.hidden_dim),
                torch.zeros(self.layer_num, self.batch_size, self.hidden_dim))

    def forward(self, vec_seq, len_seq):
        """
        :param vec_seq: tensor max_len x vec_dim with padding
        :param len_seq: tensor max_len
        :return: lstm out
        """
        # sort
        _, index = torch.sort(len_seq, descending=True)
        _, unsort_index = torch.sort(index, descending=False)
        vec_seq = vec_seq[index]
        len_seq = len_seq[index]

        # pack and pad
        vec_seq = pack_padded_sequence(vec_seq, len_seq.data.numpy(), batch_first=True)
        lstm_out, _ = self.lstm(vec_seq)
        # lstm_out, self.hidden = self.lstm(vec_seq, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # unsort
        lstm_out = lstm_out[unsort_index]

        return lstm_out


def proc_batch(batch_qa):
    """
    :param batch_qa: question: list, question_len: int, answer_list: 2d list, answer_len_list: int list
    """
    q = torch.cat([torch.Tensor(qa.question).long().view(1, -1) for qa in batch_qa], 0)
    len_q = torch.Tensor([qa.question_len for qa in batch_qa]).long()
    a_list = [torch.Tensor(qa.answer_list).long() for qa in batch_qa]
    len_a_list = [torch.Tensor(qa.answer_len_list).long() for qa in batch_qa]
    tag_a_list = [qa.answer_tag_list for qa in batch_qa]
    return q, len_q, a_list, len_a_list, tag_a_list


"""reuse the hidden state every time & set the batch as 1 ?"""


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, word_vec_matrix):
        super(Net, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_vec_matrix))

        self.LSTM_Q = LSTM(input_size=embedding_dim, hidden_size=hidden_size)
        self.LSTM_A = LSTM(input_size=embedding_dim, hidden_size=hidden_size)

        lstm_out_dim = 2 * hidden_size
        self.W = nn.Parameter(torch.FloatTensor(lstm_out_dim, lstm_out_dim))
        init.xavier_normal_(self.W)

    def forward(self, batch, batch_size):
        q = batch[0]
        len_q = batch[1]
        a_list = batch[2]
        len_a_list = batch[3]
        tag_a_list = batch[4]

        embed_q = self.word_embeddings(q)
        embed_a_list = [self.word_embeddings(a) for a in a_list]

        lstm_out_q = self.LSTM_Q(embed_q, len_q)  # batch_size x sent_len x dim
        lstm_out_a_list = []  # batch_size x answer_num x sent_len x dim
        for embed_a, len_a in zip(embed_a_list, len_a_list):
            lstm_out_a_list.append(self.LSTM_A(embed_a, len_a))

        loss_batch = []

        for i in range(batch_size):
            Q = lstm_out_q[i]
            As = lstm_out_a_list[i]
            A_num = As.shape[0]
            similarity = []

            for j in range(A_num):
                A = As[j]
                # Q x W x AT
                QW = Q.mm(self.W)
                QWAT = QW.mm(A.t())
                matrix = torch.tanh(QWAT)

                weight_q = torch.max(matrix, 1)[0]
                weight_a = torch.max(matrix, 0)[0]

                vector_q = weight_q.view(1, -1).mm(Q)
                vector_a = weight_a.view(1, -1).mm(A)

                # cos
                cos = F.cosine_similarity(vector_q, vector_a).view(-1)

                if tag_a_list[i][j] == 1:
                    cos = -cos

                similarity.append(cos)

            loss_cur = sum(similarity) / A_num
            loss_batch.append(loss_cur)

        loss = sum(loss_batch) / batch_size

        return loss
