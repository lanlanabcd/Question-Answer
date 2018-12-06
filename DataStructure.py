from Model import proc_batch, Net

import numpy as np

class QAElement:
    def __init__(self):
        """
        question: list, question_len: int, answer_list: 2d list, answer_len: int list
        """
        self.question = []              # a list of int, and every number represents a word
        self.question_len = 0           # length of self.question
        self.answer_list = [[]]         # of the same format with question
        self.answer_len_list = []       # length of each question
        self.answer_tag_list = []       # tag of each question

# TODO(WZL): for LXZ: initial the QAElement for each question-answers pair.


# temporary test data ONLY FOR TESTING MODEL

# qa_batch = [QAElement()]
#
# qa_batch[0].question = [1, 2, 3]
# qa_batch[0].question_len = 3
# qa_batch[0].answer_list = [[1, 2, 3]]
# qa_batch[0].answer_len_list = [3]
# qa_batch[0].answer_tag_list = [1]
#
# qa_batch = proc_batch(qa_batch)
# #vocab_size, embedding_dim, hidden_size, word_vec_matrix)
#
# m = np.zeros(shape=[4, 10])
#
# model = Net(vocab_size=4, embedding_dim=10, hidden_size=10, word_vec_matrix=m)
#
# model(qa_batch, 1)
