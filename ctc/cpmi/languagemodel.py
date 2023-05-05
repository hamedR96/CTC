
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from ctc.cpmi import task

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM


class LanguageModel:

    def __init__(self, device, model_spec, batchsize, state_dict=None):
        self.device = device
        self.model_spec = model_spec
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=model_spec,
            state_dict=state_dict).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_spec)
        self.batchsize = batchsize


    def _create_pmi_dataset(
            self, ptb_tokenlist,
            pad_left=None, pad_right=None,
            add_special_tokens=True, verbose=True):
        raise NotImplementedError

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, add_special_tokens=True,
            pad_left=None, pad_right=None, corruption=None,
            verbose=True):
        """Maps tokenlist to PMI matrix, and also returns pseudo log likelihood
        (override in implementing class)."""
        raise NotImplementedError

    def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
        raise NotImplementedError


class Baseline:
    """ Compute a linear/random baseline as if it were a model. """
    def __init__(self, baseline_spec):
        self.baseline_spec = baseline_spec


    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, add_special_tokens=True,
            pad_left=None, pad_right=None, corruption=None,
            verbose=True):
        pseudo_loglik = None
        fake_observation = [ptb_tokenlist]
        if self.baseline_spec == 'linear_baseline':
            distances = task.LinearBaselineTask.labels(fake_observation)
            fake_pmi_matrix = 1/(distances+1e-15)
            return fake_pmi_matrix, pseudo_loglik 
        elif self.baseline_spec == 'random_baseline':
            fake_pmi_matrix = task.RandomBaselineTask.labels(fake_observation)
            return fake_pmi_matrix, pseudo_loglik 



class XLNetSentenceDataset(torch.utils.data.Dataset):
    """Dataset class for XLNet"""
    def __init__(
            self, input_ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=6, n_pad_left=0, n_pad_right=0):
        self.input_ids = input_ids
        self.n_pad_left = n_pad_left
        self.n_pad_right = n_pad_right
        self.mask_token_id = mask_token_id
        self.ptbtok_to_span = ptbtok_to_span
        self.span_to_ptbtok = span_to_ptbtok
        self._make_tasks()

    @staticmethod
    def collate_fn(batch):
        """concatenate and prepare batch"""
        tbatch = {}
        tbatch["input_ids"] = torch.LongTensor(np.array([b['input_ids'] for b in batch]))
        tbatch["perm_mask"] = torch.FloatTensor(np.array([b['perm_mask'] for b in batch]))
        tbatch["target_map"] = torch.FloatTensor(np.array([b['target_map'] for b in batch]))
        tbatch["target_id"] = [b['target_id'] for b in batch]
        tbatch["source_span"] = [b['source_span'] for b in batch]
        tbatch["target_span"] = [b['target_span'] for b in batch]
        return tbatch

    def _make_tasks(self):
        tasks = []
        len_s = len(self.input_ids)  # length in subword tokens
        len_t = len(self.ptbtok_to_span)  # length in ptb tokens
        for source_span in self.ptbtok_to_span:
            for target_span in self.ptbtok_to_span:
                for idx_target, target_pos in enumerate(target_span):
                    # these are the positions of the source span
                    abs_source = [self.n_pad_left + s for s in source_span]
                    # this is the token we want to predict in the target span
                    abs_target_curr = self.n_pad_left + target_pos
                    # these are all the tokens we need to mask in the target span
                    abs_target_next = [self.n_pad_left + t
                                       for t in target_span[idx_target:]]
                    # we replace all hidden target tokens with <mask>
                    input_ids = np.array(self.input_ids)
                    input_ids[abs_target_next] = self.mask_token_id
                    # create permutation mask
                    perm_mask = np.zeros((len_s, len_s))
                    perm_mask[:, abs_target_next] = 1.
                    # if the source span is different from target span,
                    # then we need to mask all of its tokens
                    if source_span != target_span:
                        input_ids[abs_source] = self.mask_token_id
                        perm_mask[:, abs_source] = 1.
                    # build prediction map
                    target_map = np.zeros((1, len_s))
                    target_map[0, abs_target_curr] = 1.
                    # build all
                    task_dict = {}
                    task_dict["input_ids"] = input_ids
                    task_dict["source_span"] = source_span
                    task_dict["target_span"] = target_span
                    task_dict["target_map"] = target_map
                    task_dict["perm_mask"] = perm_mask
                    task_dict["target_id"] = self.input_ids[abs_target_curr]
                    tasks.append(task_dict)
        self._tasks = tasks

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, idx):
        return self._tasks[idx]


class XLNet(LanguageModel):
    """Class for using XLNet as estimator"""

    def __init__(self, device, model_spec, batchsize, state_dict=None):
            self.device = device
            self.model_spec = model_spec
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_spec,
                state_dict=state_dict).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_spec)
            self.batchsize = batchsize


    def _create_pmi_dataset(
            self, ptb_tokenlist,
            pad_left=None, pad_right=None,
            add_special_tokens=True, verbose=True):

        # map each ptb token to a list of spans
        # [0, 1, 2] -> [(0,), (1, 2,), (3,)]
        tokens, ptbtok_to_span = self.make_subword_lists(
            ptb_tokenlist, add_special_tokens=False)

        # map each span to the ptb token position
        # {(0,): 0, (1, 2,): 1, (3,): 2}
        span_to_ptbtok = {}
        for i, span in enumerate(ptbtok_to_span):
            assert span not in span_to_ptbtok
            span_to_ptbtok[span] = i

        # just convert here, tokenization is taken care of by make_subword_lists
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # add special characters add optional padding
        if pad_left:
            pad_left_tokens, _ = self.make_subword_lists(pad_left)
            pad_left = self.tokenizer.convert_tokens_to_ids(pad_left_tokens)
            if add_special_tokens:
                pad_left += [self.tokenizer.sep_token_id]
        else:
            pad_left = []
        if pad_right:
            pad_right_tokens, _ = self.make_subword_lists(pad_right)
            pad_right = self.tokenizer.convert_tokens_to_ids(pad_right_tokens)
        else:
            pad_right = []
        if add_special_tokens:
            pad_right += [self.tokenizer.sep_token_id,
                          self.tokenizer.cls_token_id]
        ids = pad_left + ids + pad_right
        n_pad_left = len(pad_left)
        n_pad_right = len(pad_right)


        # setup data loader
        dataset = XLNetSentenceDataset(
            ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=self.tokenizer.mask_token_id,
            n_pad_left=n_pad_left, n_pad_right=n_pad_right)
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=self.batchsize,
            collate_fn=XLNetSentenceDataset.collate_fn)
        return dataset, loader

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, add_special_tokens=True,
            pad_left=None, pad_right=None, corruption=None,
            verbose=True):

        # create dataset for observed ptb sentence
        dataset, loader = self._create_pmi_dataset(
            ptb_tokenlist, verbose=verbose,
            pad_left=pad_left, pad_right=pad_right,
            add_special_tokens=add_special_tokens)

        # use model to compute PMIs
        results = []
        for batch in loader:
            outputs = self.model(
                batch['input_ids'].to(self.device),
                perm_mask=batch['perm_mask'].to(self.device),
                target_mapping=batch['target_map'].to(self.device))
            outputs = F.log_softmax(outputs[0], 2)
            for i, output in enumerate(outputs):
                # the token id we need to predict, this belongs to target span
                target_id = batch['target_id'][i]
                assert output.size(0) == 1
                log_target = output[0, target_id].item()
                result_dict = {}
                result_dict['source_span'] = batch['source_span'][i]
                result_dict['target_span'] = batch['target_span'][i]
                result_dict['log_target'] = log_target
                result_dict['target_id'] = target_id
                results.append(result_dict)

        num_ptbtokens = len(ptb_tokenlist)
        log_p = np.zeros((num_ptbtokens, num_ptbtokens))
        # num = np.zeros((num_ptbtokens, num_ptbtokens))
        for result in results:
            log_target = result['log_target']
            source_span = result['source_span']
            target_span = result['target_span']
            ptbtok_source = dataset.span_to_ptbtok[source_span]
            ptbtok_target = dataset.span_to_ptbtok[target_span]
            if len(target_span) == 1:
                # sanity check: if target_span is 1 token, then we don't need
                # to accumulate subwords probabilities
                assert log_p[ptbtok_target, ptbtok_source] == 0.
            # we accumulate all log probs for subwords in a given span
            log_p[ptbtok_target, ptbtok_source] += log_target
            # num[ptbtok_target, ptbtok_source] += 1
        # tqdm.write(f'num:\n{num}')

        # PMI(w_i, w_j | c ) = log p(w_i | c) - log p(w_i | c \ w_j)
        # log_p[i, i] is log p(w_i | c)
        # log_p[i, j] is log p(w_i | c \ w_j)
        log_p_wi_I_c = np.diag(log_p)
        pseudo_loglik = np.trace(log_p)
        pmi_matrix = log_p_wi_I_c[:, None] - log_p
        return pmi_matrix, pseudo_loglik

    def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
        '''
        Takes list of items from Penn Treebank tokenized text,
        runs the tokenizer to decompose into the subword tokens expected by XLNet,
        including appending special characters '<sep>' and '<cls>', if specified.
        Implements some simple custom adjustments to make the results more like what might be expected.
        [TODO: this could be improved, if it is important.
        For instance, currently it puts an extra space before opening quotes]
        Returns:
            tokens: a flat list of subword tokens
            ptbtok_to_span: a list of tuples, of length = len(ptb_tokenlist <+ special tokens>)
                where the nth tuple is token indices for the nth ptb word. TODO
        '''
        subword_lists = []
        for word in ptb_tokenlist:
            if word == '-LCB-': word = '{'
            elif word == '-RCB-': word = '}'
            elif word == '-LSB-': word = '['
            elif word == '-RSB-': word = ']'
            elif word == '-LRB-': word = '('
            elif word == '-RRB-': word = ')'
            word_tokens = self.tokenizer.tokenize(word)
            subword_lists.append(word_tokens)
        if add_special_tokens:
            subword_lists.append(['<sep>'])
            subword_lists.append(['<cls>'])
        # Custom adjustments below
        for i, subword_list_i in enumerate(subword_lists):
            if subword_list_i[0][0] == '▁' and subword_lists[i-1][-1] in ('(','[','{'):
                # tqdm.write(f'{i}: removing extra space after character. {subword_list_i[0]} => {subword_list_i[0][1:]}')
                subword_list_i[0] = subword_list_i[0][1:]
                if subword_list_i[0] == '':
                    subword_list_i.pop(0)
            if subword_list_i[0] == '▁' and subword_list_i[1] in (')',']','}',',','.','"',"'","!","?") and i != 0:
                # tqdm.write(f'{i}: removing extra space before character. {subword_list_i} => {subword_list_i[1:]}')
                subword_list_i.pop(0)
            if subword_list_i == ['▁', 'n', "'", 't'] and i != 0:
                # tqdm.write(f"{i}: fixing X▁n't => Xn 't ")
                del subword_list_i[0]
                del subword_list_i[0]
                subword_lists[i-1][-1] += 'n'

        tokens = list(itertools.chain(*subword_lists)) # flattened list
        ptbtok_to_span = []
        pos = 0
        for token in subword_lists:
            ptbtok_to_span.append(())
            for _ in token:
                ptbtok_to_span[-1] = ptbtok_to_span[-1] + (pos,)
                pos += 1
        return tokens, ptbtok_to_span


class BERTSentenceDataset(torch.utils.data.Dataset):
    """Dataset class for BERT"""

    def __init__(
            self, input_ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=103, n_pad_left=0, n_pad_right=0,
            corruption=None):
        self.input_ids = input_ids
        self.n_pad_left = n_pad_left
        self.n_pad_right = n_pad_right
        self.mask_token_id = mask_token_id
        self.ptbtok_to_span = ptbtok_to_span
        self.span_to_ptbtok = span_to_ptbtok
        self.corruption = corruption
        self._make_tasks()

    @staticmethod
    def collate_fn(batch):
        """concatenate and prepare batch"""
        tbatch = {}
        tbatch["input_ids"] = torch.LongTensor(np.array([b['input_ids'] for b in batch]))
        tbatch["target_loc"] = [b['target_loc'] for b in batch]
        tbatch["target_id"] = [b['target_id'] for b in batch]
        tbatch["source_span"] = [b['source_span'] for b in batch]
        tbatch["target_span"] = [b['target_span'] for b in batch]
        return tbatch

    def _make_tasks(self):
        tasks = []
        for source_span in self.ptbtok_to_span:
            for target_span in self.ptbtok_to_span:
                if self.corruption == 'random_masking':
                    sentence_length = len(self.input_ids) - self.n_pad_left - self.n_pad_right
                    target_and_source_locs = source_span + target_span
                    possible_corrupt_locs = np.array([
                        i for i in range(sentence_length)
                        if i not in target_and_source_locs]) + self.n_pad_left
                    whether_to_corrupt = np.random.binomial(1, 30/100, len(possible_corrupt_locs)).astype(bool)
                    corrupt_locs = possible_corrupt_locs[whether_to_corrupt]
                    # tqdm.write(f'=== doing masking. target_span+source_span:{target_and_source_locs}')
                    # tqdm.write(f'=== possible_corrupt_locs: {possible_corrupt_locs}')
                    # tqdm.write(f'=== corrupt_locs: {corrupt_locs}')
                for idx_target, target_pos in enumerate(target_span):
                    # these are the positions of the source span
                    abs_source = [self.n_pad_left + s for s in source_span]
                    # this is the token we want to predict in the target span
                    abs_target_curr = self.n_pad_left + target_pos
                    # these are all the tokens we need to mask in the target span
                    abs_target_next = [self.n_pad_left + t
                                       for t in target_span[idx_target:]]
                    # we replace all hidden target tokens with [MASK]
                    input_ids = np.array(self.input_ids)
                    input_ids[abs_target_next] = self.mask_token_id
                    if self.corruption == 'random_masking':
                        input_ids[corrupt_locs] = self.mask_token_id
                    # if the source span is different from target span,
                    # then we need to mask all of its tokens
                    if source_span != target_span:
                        input_ids[abs_source] = self.mask_token_id
                    # the loc in input list to predict (bc bert predicts all)
                    target_loc = abs_target_curr
                    # build all
                    task_dict = {}
                    task_dict["input_ids"] = input_ids
                    task_dict["source_span"] = source_span
                    task_dict["target_span"] = target_span
                    task_dict["target_loc"] = target_loc
                    task_dict["target_id"] = self.input_ids[abs_target_curr]
                    tasks.append(task_dict)
        self._tasks = tasks

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, idx):
        return self._tasks[idx]


class BERT(LanguageModel):
    """Class for using BERT as estimator"""

    def _create_pmi_dataset(
            self, ptb_tokenlist,
            pad_left=None, pad_right=None,
            add_special_tokens=True, verbose=True,
            corruption=None):

        # map each ptb token to a list of spans
        # [0, 1, 2] -> [(0,), (1, 2,), (3,)]
        tokens, ptbtok_to_span = self.make_subword_lists(
            ptb_tokenlist, add_special_tokens=False)

        # map each span to the ptb token position
        # {(0,): 0, (1, 2,): 1, (3,): 2}
        span_to_ptbtok = {}
        for i, span in enumerate(ptbtok_to_span):
            assert span not in span_to_ptbtok
            span_to_ptbtok[span] = i

        # just convert here, tokenization is taken care of by make_subword_lists
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # add special characters add optional padding
        if pad_left:
            pad_left_tokens, _ = self.make_subword_lists(pad_left)
            if add_special_tokens:
                pad_left = [self.tokenizer.cls_token_id]
            pad_left += self.tokenizer.convert_tokens_to_ids(pad_left_tokens)
            if add_special_tokens:
                pad_left += [self.tokenizer.sep_token_id]
        else:
            pad_left = [self.tokenizer.cls_token_id]
        if pad_right:
            pad_right_tokens, _ = self.make_subword_lists(pad_right)
            pad_right = self.tokenizer.convert_tokens_to_ids(pad_right_tokens)
        else:
            pad_right = []
        if add_special_tokens:
            pad_right += [self.tokenizer.sep_token_id]
        ids = pad_left + ids + pad_right
        n_pad_left = len(pad_left)
        n_pad_right = len(pad_right)



        # setup data loader
        dataset = BERTSentenceDataset(
            ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=self.tokenizer.mask_token_id,
            n_pad_left=n_pad_left, n_pad_right=n_pad_right,
            corruption=corruption)
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=self.batchsize,
            collate_fn=BERTSentenceDataset.collate_fn)
        return dataset, loader

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, add_special_tokens=True,
            pad_left=None, pad_right=None, corruption=None,
            verbose=True):
        '''
        input: ptb_tokenlist: PTB-tokenized sentence as list
        return: pmi matrix for that sentence
        '''

        # create dataset for observed ptb sentence
        dataset, loader = self._create_pmi_dataset(
            ptb_tokenlist, verbose=verbose,
            pad_left=pad_left, pad_right=pad_right,
            add_special_tokens=add_special_tokens,
            corruption=corruption)

        # use model to compute PMIs
        results = []
        for batch in loader:
            outputs = self.model(
                batch['input_ids'].to(self.device))
            outputs = F.log_softmax(outputs[0], 2)
            for i, output in enumerate(outputs):
                # the token id we need to predict, this belongs to target span
                target_id = batch['target_id'][i]
                input_ids = batch['input_ids'][i]
                target_loc = batch['target_loc'][i]
                assert output.size(0) == len(input_ids)
                log_target = output[target_loc, target_id].item()
                result_dict = {}
                result_dict['source_span'] = batch['source_span'][i]
                result_dict['target_span'] = batch['target_span'][i]
                result_dict['log_target'] = log_target
                result_dict['target_id'] = target_id
                results.append(result_dict)

        num_ptbtokens = len(ptb_tokenlist)
        log_p = np.zeros((num_ptbtokens, num_ptbtokens))
        # num = np.zeros((num_ptbtokens, num_ptbtokens))
        for result in results:
            log_target = result['log_target']
            source_span = result['source_span']
            target_span = result['target_span']
            ptbtok_source = dataset.span_to_ptbtok[source_span]
            ptbtok_target = dataset.span_to_ptbtok[target_span]
            if len(target_span) == 1:
                # sanity check: if target_span is 1 token, then we don't need
                # to accumulate subwords probabilities
                assert log_p[ptbtok_target, ptbtok_source] == 0.
            # we accumulate all log probs for subwords in a given span
            log_p[ptbtok_target, ptbtok_source] += log_target
            # num[ptbtok_target, ptbtok_source] += 1
        # tqdm.write(f'num:\n{num}')

        # PMI(w_i, w_j | c ) = log p(w_i | c) - log p(w_i | c \ w_j)
        # log_p[i, i] is log p(w_i | c)
        # log_p[i, j] is log p(w_i | c \ w_j)
        log_p_wi_I_c = np.diag(log_p)
        pseudo_loglik = np.trace(log_p)
        pmi_matrix = log_p_wi_I_c[:, None] - log_p
        return pmi_matrix, pseudo_loglik

    def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
        '''
        Takes list of items from Penn Treebank tokenized text,
        runs the tokenizer to decompose into the subword tokens expected by XLNet,
        including appending special characters '[CLS]' and '[SEP]', if specified.
        Implements some simple custom adjustments to make the results more like what might be expected.
        [TODO: this could be improved, if it is important.
        For instance, currently it puts an extra space before opening quotes]
        Returns:
            tokens: a flat list of subword tokens
            ptbtok_to_span: a list of tuples, of length = len(ptb_tokenlist <+ special tokens>)
                where the nth tuple is token indices for the nth ptb word.
        '''
        subword_lists = []
        if add_special_tokens:
            subword_lists.append(['[CLS]'])
        for word in ptb_tokenlist:
            if word == '-LCB-': word = '{'
            elif word == '-RCB-': word = '}'
            elif word == '-LSB-': word = '['
            elif word == '-RSB-': word = ']'
            elif word == '-LRB-': word = '('
            elif word == '-RRB-': word = ')'
            word_tokens = self.tokenizer.tokenize(word)
            subword_lists.append(word_tokens)
        if add_special_tokens:
            subword_lists.append(['[SEP]'])
        # Custom adjustments below
        for i, subword_list_i in enumerate(subword_lists):
            if subword_list_i == ['n', "'", 't'] and i != 0:
                # tqdm.write(f"{i}: fixing X n ' t => Xn ' t ")
                del subword_list_i[0]
                subword_lists[i-1][-1] += 'n'

        tokens = list(itertools.chain(*subword_lists)) # flattened list
        ptbtok_to_span = []
        pos = 0
        for token in subword_lists:
            ptbtok_to_span.append(())
            for _ in token:
                ptbtok_to_span[-1] = ptbtok_to_span[-1] + (pos,)
                pos += 1
        return tokens, ptbtok_to_span


class BartSentenceDataset(torch.utils.data.Dataset):
    """Dataset class for Bart"""

    def __init__(
            self, input_ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=50264, n_pad_left=0, n_pad_right=0):
        self.input_ids = input_ids
        self.n_pad_left = n_pad_left
        self.n_pad_right = n_pad_right
        self.mask_token_id = mask_token_id
        self.ptbtok_to_span = ptbtok_to_span
        self.span_to_ptbtok = span_to_ptbtok
        self._make_tasks()

    @staticmethod
    def collate_fn(batch):
        """concatenate and prepare batch"""
        tbatch = {}
        tbatch["input_ids"] = torch.LongTensor(np.array([b['input_ids'] for b in batch]))
        tbatch["target_loc"] = [b['target_loc'] for b in batch]
        tbatch["target_id"] = [b['target_id'] for b in batch]
        tbatch["source_span"] = [b['source_span'] for b in batch]
        tbatch["target_span"] = [b['target_span'] for b in batch]
        return tbatch

    def _make_tasks(self):
        tasks = []
        for source_span in self.ptbtok_to_span:
            for target_span in self.ptbtok_to_span:
                for idx_target, target_pos in enumerate(target_span):
                    # these are the positions of the source span
                    abs_source = [self.n_pad_left + s for s in source_span]
                    # this is the token we want to predict in the target span
                    abs_target_curr = self.n_pad_left + target_pos
                    # these are all the tokens we need to mask in the target span
                    abs_target_next = [self.n_pad_left + t
                                       for t in target_span[idx_target:]]
                    # we replace all hidden target tokens with [MASK]
                    input_ids = np.array(self.input_ids)
                    input_ids[abs_target_next] = self.mask_token_id
                    # if the source span is different from target span,
                    # then we need to mask all of its tokens
                    if source_span != target_span:
                        input_ids[abs_source] = self.mask_token_id
                    # the location in the input list to predict (since bert predicts all)
                    target_loc = abs_target_curr
                    # build all
                    task_dict = {}
                    task_dict["input_ids"] = input_ids
                    task_dict["source_span"] = source_span
                    task_dict["target_span"] = target_span
                    task_dict["target_loc"] = target_loc
                    task_dict["target_id"] = self.input_ids[abs_target_curr]
                    tasks.append(task_dict)
        self._tasks = tasks

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, idx):
        return self._tasks[idx]


class Bart(LanguageModel):
    """Class for using Bart as estimator"""

    def _create_pmi_dataset(
            self, ptb_tokenlist,
            pad_left=None, pad_right=None,
            add_special_tokens=True, verbose=True):

        # map each ptb token to a list of spans
        # [0, 1, 2] -> [(0,), (1, 2,), (3,)]
        tokens, ptbtok_to_span = self.make_subword_lists(
            ptb_tokenlist, add_special_tokens=False)

        # map each span to the ptb token position
        # {(0,): 0, (1, 2,): 1, (3,): 2}
        span_to_ptbtok = {}
        for i, span in enumerate(ptbtok_to_span):
            assert span not in span_to_ptbtok
            span_to_ptbtok[span] = i

        # just convert here, tokenization is taken care of by make_subword_lists
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # add special characters add optional padding
        if pad_left:
            pad_left_tokens, _ = self.make_subword_lists(pad_left)
            if add_special_tokens:
                pad_left = [self.tokenizer.bos_token_id]
            pad_left += self.tokenizer.convert_tokens_to_ids(pad_left_tokens)
            # if add_special_tokens:
            #   pad_left += [self.tokenizer.sep_token_id]
        else:
            pad_left = [self.tokenizer.bos_token_id]
        if pad_right:
            pad_right_tokens, _ = self.make_subword_lists(pad_right)
            pad_right = self.tokenizer.convert_tokens_to_ids(pad_right_tokens)
        else:
            pad_right = []
        if add_special_tokens:
            pad_right += [self.tokenizer.eos_token_id]
        ids = pad_left + ids + pad_right
        n_pad_left = len(pad_left)
        n_pad_right = len(pad_right)


        # setup data loader
        dataset = BartSentenceDataset(
            ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=self.tokenizer.mask_token_id,
            n_pad_left=n_pad_left, n_pad_right=n_pad_right)
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=self.batchsize,
            collate_fn=BartSentenceDataset.collate_fn)
        return dataset, loader

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, add_special_tokens=True,
            pad_left=None, pad_right=None, corruption=None,
            verbose=True):
        '''
        input: ptb_tokenlist: PTB-tokenized sentence as list
        return: pmi matrix for that sentence
        '''

        # create dataset for observed ptb sentence
        dataset, loader = self._create_pmi_dataset(
            ptb_tokenlist, verbose=verbose,
            pad_left=pad_left, pad_right=pad_right,
            add_special_tokens=add_special_tokens)

        # use model to compute PMIs
        results = []
        for batch in loader:
            outputs = self.model(
                batch['input_ids'].to(self.device))
            outputs = F.log_softmax(outputs[0], 2)
            for i, output in enumerate(outputs):
                # the token id we need to predict, this belongs to target span
                target_id = batch['target_id'][i]
                input_ids = batch['input_ids'][i]
                target_loc = batch['target_loc'][i]
                assert output.size(0) == len(input_ids)
                log_target = output[target_loc, target_id].item()
                result_dict = {}
                result_dict['source_span'] = batch['source_span'][i]
                result_dict['target_span'] = batch['target_span'][i]
                result_dict['log_target'] = log_target
                result_dict['target_id'] = target_id
                results.append(result_dict)

        num_ptbtokens = len(ptb_tokenlist)
        log_p = np.zeros((num_ptbtokens, num_ptbtokens))
        # num = np.zeros((num_ptbtokens, num_ptbtokens))
        for result in results:
            log_target = result['log_target']
            source_span = result['source_span']
            target_span = result['target_span']
            ptbtok_source = dataset.span_to_ptbtok[source_span]
            ptbtok_target = dataset.span_to_ptbtok[target_span]
            if len(target_span) == 1:
                # sanity check: if target_span is 1 token, then we don't need
                # to accumulate subwords probabilities
                assert log_p[ptbtok_target, ptbtok_source] == 0.
            # we accumulate all log probs for subwords in a given span
            log_p[ptbtok_target, ptbtok_source] += log_target
            # num[ptbtok_target, ptbtok_source] += 1
        # tqdm.write(f'num:\n{num}')

        # PMI(w_i, w_j | c ) = log p(w_i | c) - log p(w_i | c \ w_j)
        # log_p[i, i] is log p(w_i | c)
        # log_p[i, j] is log p(w_i | c \ w_j)
        log_p_wi_I_c = np.diag(log_p)
        pseudo_loglik = np.trace(log_p)
        pmi_matrix = log_p_wi_I_c[:, None] - log_p
        return pmi_matrix, pseudo_loglik

    def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
        '''
        Takes list of items from Penn Treebank tokenized text,
        runs the tokenizer to decompose into the subword tokens expected from BPE for Bart,
        Implements some simple custom adjustments to make the results more like what might be expected.
        [TODO: this could be improved, if it is important.]
        Returns:
            tokens: a flat list of subword tokens
            ptbtok_to_span: a list of tuples, of length = len(ptb_tokenlist <+ special tokens>)
                where the nth tuple is token indices for the nth ptb word.
        '''
        subword_lists = []
        if add_special_tokens:
            subword_lists.append(['<s>'])
        for index, word in enumerate(ptb_tokenlist):
            if word == '-LCB-': word = '{'
            elif word == '-RCB-': word = '}'
            elif word == '-LSB-': word = '['
            elif word == '-RSB-': word = ']'
            elif word == '-LRB-': word = '('
            elif word == '-RRB-': word = ')'
            add_prefix_space = True
            if word[0] in [',','.',':',';','!',"'",")","]","}"]:
                add_prefix_space = False
            if index > 0 and ptb_tokenlist[index-1] in ["(","[","{","`","``"]:
                add_prefix_space = False
            word_tokens = self.tokenizer.tokenize(word, add_prefix_space=add_prefix_space)
            subword_lists.append(word_tokens)
        if add_special_tokens:
            subword_lists.append(['</s>'])
        # Custom adjustments below
        for i, subword_list_i in enumerate(subword_lists):
            if subword_list_i == ['\u0120n', "'t"] and i != 0:
                # tqdm.write(f"{i}: fixing X n 't => Xn 't ")
                del subword_list_i[0]
                subword_lists[i-1][-1] += 'n'

        tokens = list(itertools.chain(*subword_lists)) # flattened list
        ptbtok_to_span = []
        pos = 0
        for token in subword_lists:
            ptbtok_to_span.append(())
            for _ in token:
                ptbtok_to_span[-1] = ptbtok_to_span[-1] + (pos,)
                pos += 1
        return tokens, ptbtok_to_span


class XLMSentenceDataset(torch.utils.data.Dataset):
    """Dataset class for XLM"""

    def __init__(
        self, input_ids, ptbtok_to_span, span_to_ptbtok,
        mask_token_id=5, n_pad_left=0, n_pad_right=0):
        self.input_ids = input_ids
        self.n_pad_left = n_pad_left
        self.n_pad_right = n_pad_right
        self.mask_token_id = mask_token_id
        self.ptbtok_to_span = ptbtok_to_span
        self.span_to_ptbtok = span_to_ptbtok
        self._make_tasks()

    @staticmethod
    def collate_fn(batch):
        """concatenate and prepare batch"""
        tbatch = {}
        tbatch["input_ids"] = torch.LongTensor(np.array([b['input_ids'] for b in batch]))
        tbatch["target_loc"] = [b['target_loc'] for b in batch]
        tbatch["target_id"] = [b['target_id'] for b in batch]
        tbatch["source_span"] = [b['source_span'] for b in batch]
        tbatch["target_span"] = [b['target_span'] for b in batch]
        return tbatch

    def _make_tasks(self):
        tasks = []
        for source_span in self.ptbtok_to_span:
            for target_span in self.ptbtok_to_span:
                for idx_target, target_pos in enumerate(target_span):
                    # these are the positions of the source span
                    abs_source = [self.n_pad_left + s for s in source_span]
                    # this is the token we want to predict in the target span
                    abs_target_curr = self.n_pad_left + target_pos
                    # these are all the tokens we need to mask in the target span
                    abs_target_next = [self.n_pad_left + t
                                       for t in target_span[idx_target:]]
                    # we replace all hidden target tokens with the mask token <special1>
                    input_ids = np.array(self.input_ids)
                    input_ids[abs_target_next] = self.mask_token_id
                    # if the source span is different from target span,
                    # then we need to mask all of its tokens
                    if source_span != target_span:
                        input_ids[abs_source] = self.mask_token_id
                    # the location in the input list to predict (since bert predicts all)
                    target_loc = abs_target_curr
                    # build all
                    task_dict = {}
                    task_dict["input_ids"] = input_ids
                    task_dict["source_span"] = source_span
                    task_dict["target_span"] = target_span
                    task_dict["target_loc"] = target_loc
                    task_dict["target_id"] = self.input_ids[abs_target_curr]
                    tasks.append(task_dict)
        self._tasks = tasks

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, idx):
        return self._tasks[idx]


class XLM(LanguageModel):
    """Class for using XLM as estimator"""

    def _create_pmi_dataset(
            self, ptb_tokenlist,
            pad_left=None, pad_right=None,
            add_special_tokens=True, verbose=True):

        # map each ptb token to a list of spans
        # [0, 1, 2] -> [(0,), (1, 2,), (3,)]
        tokens, ptbtok_to_span = self.make_subword_lists(
            ptb_tokenlist, add_special_tokens=False)

        # map each span to the ptb token position
        # {(0,): 0, (1, 2,): 1, (3,): 2}
        span_to_ptbtok = {}
        for i, span in enumerate(ptbtok_to_span):
            assert span not in span_to_ptbtok
            span_to_ptbtok[span] = i

        # just convert here, tokenization is taken care of by make_subword_lists
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # add special characters add optional padding
        if pad_left:
            pad_left_tokens, _ = self.make_subword_lists(pad_left)
            if add_special_tokens:
                pad_left = [self.tokenizer.cls_token_id] # cls token is </s>
            pad_left += self.tokenizer.convert_tokens_to_ids(pad_left_tokens)
            if add_special_tokens:
                pad_left += [self.tokenizer.sep_token_id] # sep token is also </s>
        else:
            pad_left = [self.tokenizer.cls_token_id]
        if pad_right:
            pad_right_tokens, _ = self.make_subword_lists(pad_right)
            pad_right = self.tokenizer.convert_tokens_to_ids(pad_right_tokens)
        else:
            pad_right = []
        if add_special_tokens:
            pad_right += [self.tokenizer.sep_token_id]
        ids = pad_left + ids + pad_right
        n_pad_left = len(pad_left)
        n_pad_right = len(pad_right)


        # setup data loader
        dataset = XLMSentenceDataset(
            ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=self.tokenizer.mask_token_id,
            n_pad_left=n_pad_left, n_pad_right=n_pad_right)
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=self.batchsize,
            collate_fn=XLMSentenceDataset.collate_fn)
        return dataset, loader

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, add_special_tokens=True,
            pad_left=None, pad_right=None, corruption=None,
            verbose=True):
        '''
        input: ptb_tokenlist: PTB-tokenized sentence as list
        return: pmi matrix for that sentence
        '''

        # create dataset for observed ptb sentence
        dataset, loader = self._create_pmi_dataset(
            ptb_tokenlist, verbose=verbose,
            pad_left=pad_left, pad_right=pad_right,
            add_special_tokens=add_special_tokens)

        # use model to compute PMIs
        results = []
        for batch in loader:
            outputs = self.model(
                batch['input_ids'].to(self.device))
            outputs = F.log_softmax(outputs[0], 2)
            for i, output in enumerate(outputs):
                # the token id we need to predict, this belongs to target span
                target_id = batch['target_id'][i]
                input_ids = batch['input_ids'][i]
                target_loc = batch['target_loc'][i]
                assert output.size(0) == len(input_ids)
                log_target = output[target_loc, target_id].item()
                result_dict = {}
                result_dict['source_span'] = batch['source_span'][i]
                result_dict['target_span'] = batch['target_span'][i]
                result_dict['log_target'] = log_target
                result_dict['target_id'] = target_id
                results.append(result_dict)

        num_ptbtokens = len(ptb_tokenlist)
        log_p = np.zeros((num_ptbtokens, num_ptbtokens))
        # num = np.zeros((num_ptbtokens, num_ptbtokens))
        for result in results:
            log_target = result['log_target']
            source_span = result['source_span']
            target_span = result['target_span']
            ptbtok_source = dataset.span_to_ptbtok[source_span]
            ptbtok_target = dataset.span_to_ptbtok[target_span]
            if len(target_span) == 1:
                # sanity check: if target_span is 1 token, then we don't need
                # to accumulate subwords probabilities
                assert log_p[ptbtok_target, ptbtok_source] == 0.
            # we accumulate all log probs for subwords in a given span
            log_p[ptbtok_target, ptbtok_source] += log_target
            # num[ptbtok_target, ptbtok_source] += 1
        # tqdm.write(f'num:\n{num}')

        # PMI(w_i, w_j | c ) = log p(w_i | c) - log p(w_i | c \ w_j)
        # log_p[i, i] is log p(w_i | c)
        # log_p[i, j] is log p(w_i | c \ w_j)
        log_p_wi_I_c = np.diag(log_p)
        pseudo_loglik = np.trace(log_p)
        pmi_matrix = log_p_wi_I_c[:, None] - log_p
        return pmi_matrix, pseudo_loglik

    def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
        '''
        Takes list of items from Penn Treebank tokenized text,
        runs the tokenizer to decompose into the subword tokens expected by XLNet,
        including appending special character '</s>' before and after, if specified.
        Implements some simple custom adjustments to make the results more like what might be expected.
        [TODO: this could be improved, if it is important.]
        Returns:
            tokens: a flat list of subword tokens
            ptbtok_to_span: a list of tuples, of length = len(ptb_tokenlist <+ special tokens>)
                where the nth tuple is token indices for the nth ptb word.
        '''
        subword_lists = []
        if add_special_tokens:
            subword_lists.append(['</s>'])
        for word in ptb_tokenlist:
            if word == '-LCB-': word = '{'
            elif word == '-RCB-': word = '}'
            elif word == '-LSB-': word = '['
            elif word == '-RSB-': word = ']'
            elif word == '-LRB-': word = '('
            elif word == '-RRB-': word = ')'
            word_tokens = self.tokenizer.tokenize(word)
            subword_lists.append(word_tokens)
        if add_special_tokens:
            subword_lists.append(['</s>'])
        # Custom adjustments below
        for i, subword_list_i in enumerate(subword_lists):
            if subword_list_i == ['n</w>', "'t</w>"] and i != 0:
                # tqdm.write(f"{i}: fixing X n 't => Xn 't ")
                del subword_list_i[0]
                subword_lists[i-1][-1] = subword_lists[i-1][-1][:-4] + 'n</w>'

        tokens = list(itertools.chain(*subword_lists)) # flattened list
        ptbtok_to_span = []
        pos = 0
        for token in subword_lists:
            ptbtok_to_span.append(())
            for _ in token:
                ptbtok_to_span[-1] = ptbtok_to_span[-1] + (pos,)
                pos += 1
        return tokens, ptbtok_to_span


class GPT2SentenceDataset(torch.utils.data.Dataset):
    """Dataset class for GPT2. Warning:
    Not bidirectional, so only lower triangular will be interpretable
    Also, masking doesn't match training.  Mask token will not be used.
    Attention masking used instead (still not like training but less catastrophic)."""

    def __init__(
            self, input_ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=50256, n_pad_left=0, n_pad_right=0):
        self.input_ids = input_ids
        self.n_pad_left = n_pad_left
        self.n_pad_right = n_pad_right
        self.mask_token_id = mask_token_id
        self.ptbtok_to_span = ptbtok_to_span
        self.span_to_ptbtok = span_to_ptbtok
        self._make_tasks()

    @staticmethod
    def collate_fn(batch):
        """concatenate and prepare batch"""
        tbatch = {}
        tbatch["input_ids"] = torch.LongTensor(np.array([b['input_ids'] for b in batch]))
        tbatch["attention_mask"] = torch.FloatTensor(
            np.array([b['attention_mask'] for b in batch]))
        tbatch["target_loc"] = [b['target_loc'] for b in batch]
        tbatch["target_id"] = [b['target_id'] for b in batch]
        tbatch["source_span"] = [b['source_span'] for b in batch]
        tbatch["target_span"] = [b['target_span'] for b in batch]
        return tbatch

    def _make_tasks(self):
        tasks = []
        len_s = len(self.input_ids)  # length in subword tokens
        for source_span in self.ptbtok_to_span:
            for target_span in self.ptbtok_to_span:
                for idx_target, target_pos in enumerate(target_span):
                    # these are the positions of the source span
                    abs_source = [self.n_pad_left + s for s in source_span]
                    # this is the token we want to predict in the target span
                    abs_target_curr = self.n_pad_left + target_pos
                    input_ids = np.array(self.input_ids)
                    # create attention_mask
                    attention_mask = np.ones((len_s))
                    attention_mask[abs_target_curr:] = 0.  # force not to attend past curr
                    # if the source span is different from target span,
                    # then we need to attn mask all of its tokens
                    if source_span != target_span:
                        attention_mask[abs_source] = 0.  # force not to attend to source
                    # the location in the input list to predict (since it predicts all)
                    target_loc = abs_target_curr - 1
                    # build all
                    task_dict = {}
                    task_dict["input_ids"] = input_ids
                    task_dict["source_span"] = source_span
                    task_dict["target_span"] = target_span
                    task_dict["target_loc"] = target_loc
                    task_dict["attention_mask"] = attention_mask
                    task_dict["target_id"] = self.input_ids[abs_target_curr]
                    tasks.append(task_dict)
        self._tasks = tasks

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, idx):
        return self._tasks[idx]


class GPT2(LanguageModel):
    """Class for using GPT2 as estimator :
    Not bidirectional, so doesn't make much sense."""

    def _create_pmi_dataset(
            self, ptb_tokenlist,
            pad_left=None, pad_right=None,
            add_special_tokens=True, verbose=True):

        # map each ptb token to a list of spans
        # [0, 1, 2] -> [(0,), (1, 2,), (3,)]
        tokens, ptbtok_to_span = self.make_subword_lists(
            ptb_tokenlist, add_special_tokens=False)

        # map each span to the ptb token position
        # {(0,): 0, (1, 2,): 1, (3,): 2}
        span_to_ptbtok = {}
        for i, span in enumerate(ptbtok_to_span):
            assert span not in span_to_ptbtok
            span_to_ptbtok[span] = i

        # just convert here, tokenization is taken care of by make_subword_lists
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # add special characters add optional padding
        if pad_left:
            pad_left_tokens, _ = self.make_subword_lists(pad_left)           # no cls token
            pad_left = [self.tokenizer.bos_token_id]
            pad_left += self.tokenizer.convert_tokens_to_ids(pad_left_tokens) # no sep token
        else:
            pad_left = [self.tokenizer.bos_token_id]  # = eos_token_id
        if pad_right:
            pad_right_tokens, _ = self.make_subword_lists(pad_right)
            pad_right = self.tokenizer.convert_tokens_to_ids(pad_right_tokens)
        else:
            pad_right = []
        if add_special_tokens:
            pad_right += [self.tokenizer.eos_token_id]
        ids = pad_left + ids + pad_right
        n_pad_left = len(pad_left)
        n_pad_right = len(pad_right)


        # setup data loader
        dataset = GPT2SentenceDataset(
            ids, ptbtok_to_span, span_to_ptbtok,
            mask_token_id=self.tokenizer.eos_token_id, # Using the only special character as MASK.
            n_pad_left=n_pad_left, n_pad_right=n_pad_right)
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=self.batchsize,
            collate_fn=GPT2SentenceDataset.collate_fn)
        return dataset, loader

    def ptb_tokenlist_to_pmi_matrix(
            self, ptb_tokenlist, add_special_tokens=True,
            pad_left=None, pad_right=None, corruption=None,
            verbose=True):
        '''
        input: ptb_tokenlist: PTB-tokenized sentence as list
        return: pmi matrix for that sentence
        '''

        # create dataset for observed ptb sentence
        dataset, loader = self._create_pmi_dataset(
            ptb_tokenlist, verbose=verbose,
            pad_left=pad_left, pad_right=pad_right,
            add_special_tokens=add_special_tokens)

        # use model to compute PMIs
        results = []
        for batch in loader:
            outputs = self.model(
                batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device))
            outputs = F.log_softmax(outputs[0], 2)
            for i, output in enumerate(outputs):
                # the token id we need to predict, this belongs to target span
                target_id = batch['target_id'][i]
                input_ids = batch['input_ids'][i]
                target_loc = batch['target_loc'][i]
                assert output.size(0) == len(input_ids)
                log_target = output[target_loc, target_id].item()
                result_dict = {}
                result_dict['source_span'] = batch['source_span'][i]
                result_dict['target_span'] = batch['target_span'][i]
                result_dict['log_target'] = log_target
                result_dict['target_id'] = target_id
                results.append(result_dict)

        num_ptbtokens = len(ptb_tokenlist)
        log_p = np.zeros((num_ptbtokens, num_ptbtokens))
        # num = np.zeros((num_ptbtokens, num_ptbtokens))
        for result in results:
            log_target = result['log_target']
            source_span = result['source_span']
            target_span = result['target_span']
            ptbtok_source = dataset.span_to_ptbtok[source_span]
            ptbtok_target = dataset.span_to_ptbtok[target_span]
            if len(target_span) == 1:
                # sanity check: if target_span is 1 token, then we don't need
                # to accumulate subwords probabilities
                assert log_p[ptbtok_target, ptbtok_source] == 0.
            # we accumulate all log probs for subwords in a given span
            log_p[ptbtok_target, ptbtok_source] += log_target
            # num[ptbtok_target, ptbtok_source] += 1
        # tqdm.write(f'num:\n{num}')

        # PMI(w_i, w_j | c ) = log p(w_i | c) - log p(w_i | c \ w_j)
        # log_p[i, i] is log p(w_i | c)
        # log_p[i, j] is log p(w_i | c \ w_j)
        log_p_wi_I_c = np.diag(log_p)
        pseudo_loglik = np.trace(log_p)
        pmi_matrix = log_p_wi_I_c[:, None] - log_p
        return pmi_matrix, pseudo_loglik

    def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
        '''
        Takes list of items from Penn Treebank tokenized text,
        runs the tokenizer to decompose into the subword tokens expected from BPE for GPT2,
        Implements some simple custom adjustments to make the results more like what might be expected.
        [TODO: this could be improved, if it is important.]
        Returns:
            tokens: a flat list of subword tokens
            ptbtok_to_span: a list of tuples, of length = len(ptb_tokenlist <+ special tokens>)
                where the nth tuple is token indices for the nth ptb word.
        '''
        subword_lists = []
        for word in ptb_tokenlist:
            add_prefix_space = False
            if word[0] not in [',','.',':',';','!',"'",]:
                add_prefix_space = True
            if word == '-LCB-': word = '{'
            elif word == '-RCB-': word = '}'
            elif word == '-LSB-': word = '['
            elif word == '-RSB-': word = ']'
            elif word == '-LRB-': word = '('
            elif word == '-RRB-': word = ')'
            word_tokens = self.tokenizer.tokenize(word, add_prefix_space=add_prefix_space)
            subword_lists.append(word_tokens)
        # Custom adjustments below
        for i, subword_list_i in enumerate(subword_lists):
            if subword_list_i == ['\u0120n', "'t"] and i != 0:
                # tqdm.write(f"{i}: fixing X n 't => Xn 't ")
                del subword_list_i[0]
                subword_lists[i-1][-1] = subword_lists[i-1][-1][:] + 'n'

        tokens = list(itertools.chain(*subword_lists)) # flattened list
        ptbtok_to_span = []
        pos = 0
        for token in subword_lists:
            ptbtok_to_span.append(())
            for _ in token:
                ptbtok_to_span[-1] = ptbtok_to_span[-1] + (pos,)
                pos += 1
        return tokens, ptbtok_to_span
