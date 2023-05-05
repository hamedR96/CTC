
import csv
import torch
import pandas as pd


class Word2Vec:

    def __init__(self, device, model_spec, path):
        self.device = device
        self.model_spec = model_spec
        with open(path) as f:
            first_line = f.readline().strip()
            vsize, dim = [int(x) for x in first_line.split(' ')]
        self.vocabsize = vsize
        self.embedding_dim = dim
        print(f'Loading word embeddings from {path} ...')
        matrix, self.word_to_index = self._load_embeddings(path)
        self.in_embedding, self.out_embedding = self._prepare_embeddings(matrix)
        print('Loaded.')
        print(f"Embedding model '{model_spec}' initialized on {device}.")

    def _load_embeddings(self, path):
        words = pd.read_csv(path, sep=" ", index_col=0, skiprows=[0],
                            na_values=None, keep_default_na=False, header=None,
                            quoting=csv.QUOTE_NONE,
                            encoding='iso-8859-1')
        if len(words.columns) == self.embedding_dim+1:
            words = words.drop(columns=[self.embedding_dim+1])
        matrix = words.values
        index_to_word = list(words[:self.vocabsize].index)
        word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }
        return matrix, word_to_index

    def _prepare_embeddings(self, matrix):
        # Add on a mean vector embedding for unks
        in_matrix = torch.tensor(matrix[:self.vocabsize])
        mean_in = torch.mean(in_matrix, 0).unsqueeze(0)
        in_matrix = torch.cat((in_matrix, mean_in))
        out_matrix = torch.tensor(matrix[self.vocabsize:])
        mean_out = torch.mean(out_matrix, 0).unsqueeze(0)
        out_matrix = torch.cat((out_matrix, mean_out))

        in_embedding = torch.nn.Embedding(
            self.vocabsize+1, self.embedding_dim).to(self.device)
        out_embedding = torch.nn.Embedding(
            self.vocabsize+1, self.embedding_dim).to(self.device)
        in_embedding.weight.data.copy_(in_matrix.to(self.device))
        out_embedding.weight.data.copy_(out_matrix.to(self.device))
        return in_embedding, out_embedding

    def _encode(self, ptb_tokenlist):
        # word_to_index = self.word_to_index
        # word_to_index["<unknown>"] = self.vocabsize
        sentence_as_ids = [
            self.word_to_index.get(word, self.vocabsize)
            for word in ptb_tokenlist]
        return sentence_as_ids

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, add_special_tokens=True,
            pad_left=None, pad_right=None, corruption=None, # signal/noise corruption not implemented
            verbose=True):
        """Maps ptb_tokenlist to PMI matrix,
        TODO: this just ignores the rest of the arguments,
        but this way no custom call from main.py"""

        sentence_as_ids = self._encode(ptb_tokenlist)
        sentence_as_ids = torch.tensor(sentence_as_ids).to(self.device)
        with torch.no_grad():
            in_sentence = self.in_embedding(sentence_as_ids)
            out_sentence = self.out_embedding(sentence_as_ids)
            pmi_matrix = torch.matmul(
                in_sentence, out_sentence.T).cpu().numpy()
        pseudo_loglik = 0  # meaningless. just for compatibility with languagemodel.py for now
        return pmi_matrix, pseudo_loglik
