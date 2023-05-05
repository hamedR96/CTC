import itertools
import torch
import string
from tqdm import tqdm
import openai
from ctc.cpmi import txt_to_pmi
import pickle


class Auto_CTC:
    def __init__(self, segments_length=15, min_segment_length=5, segment_step=10,device="cpu"):

        self.segments_length = segments_length
        self.segment_step = segment_step
        self.min_segment_length = min_segment_length
        self.DEVICE = torch.device(device)

        self.topics = None
        self.openai_key = None
        self.documents = None
        self.segments = None
        self.cpmi_score = None
        self.cpmi_topics_score = None
        self.MODEL= None
        self.cpmi_tree = None


    def segmenting_documents(self, documents):
        self.documents = documents
        sentences = [document.strip().lower() for document in self.documents]
        sentences = ["".join([char for char in text if char not in string.punctuation]) for text in sentences]
        sentences = [sentence.split(' ') for sentence in sentences]
        sentences = [[token for token in sentence if len(token) > 0] for sentence in sentences]
        sentences = [sentence for sentence in sentences if len(sentence) > self.min_segment_length]
        sentences = [sentence[i:i + self.segments_length] for sentence in sentences
                     for i in range(0, len(sentence) - self.segments_length + 1, self.segment_step)]
        self.segments = [sentence for sentence in sentences if len(sentence) > self.min_segment_length]


    def create_cpmi_tree(self, save=True):
        self.MODEL = txt_to_pmi.languagemodel.BERT(self.DEVICE, 'bert-base-cased', 32)
        self.cpmi_tree = txt_to_pmi.get_cpmi(self.MODEL, self.segments, verbose=False)[0]
        if save:
            with open("cpmi_tree.pkl", "wb") as fp:  # Pickling
                pickle.dump(self.cpmi_tree, fp)
            with open("cpmi_segments.pkl", "wb") as fp:  # Pickling
                pickle.dump(self.segments, fp)

    def load_cpmi_tree(self):
        with open("cpmi_tree.pkl", "rb") as fp:
            self.cpmi_tree = pickle.load(fp)
        with open("cpmi_segments.pkl", "rb") as fp:
            self.segments = pickle.load(fp)


    def ctc_cpmi(self,topics):
        self.topics = topics
        self.cpmi_topics_score = []
        for topic in tqdm(self.topics):
            topic_score = 0
            for pairs in list(itertools.permutations(topic, 2)):
                for i, sentence in enumerate(self.segments):
                    if pairs[0] in sentence and pairs[1] in sentence:
                            w1 = sentence.index(pairs[0])
                            w2 = sentence.index(pairs[1])
                            topic_score += self.cpmi_tree[str(i)][w1][w2] / 2
            self.cpmi_topics_score.append(topic_score)
        self.cpmi_score=sum(self.cpmi_topics_score)/len(self.cpmi_topics_score)
        return self.cpmi_score

class Semi_auto_CTC:
    def __init__(self, openai_key,topics):
        openai.api_key = openai_key
        self.topics = topics
        self.intrusion_score = None
        self.intrusion_topics_score = None
        self.rating_score = None
        self.rating_topics_score = None

    def ctc_intrusion(self):
        self.intrusion_topics_score = []
        for topic in self.topics:
            topic_words = ", ".join(topic)
            prompt = "I have a topic that is described by the following keywords: [" + topic_words + "]. Provide a one-word topic based on this list of words and identify all intruder words in the list with respect to the topic you provided in the following format: topic: <one-word> , intruders: <words in the list> without explanation."
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [{"role": "user", "content": prompt}])
            answer = response['choices'][0]['message']['content']
            answer = answer.replace('\n', ', ')
            parts = answer.split(':')
            topic = parts[1].split(' ')[1].split(",")[0]
            intruder_words = parts[2].split(',')
            self.intrusion_topics_score.append(1 - len(intruder_words)/len(topic))
        self.intrusion_score=sum(self.intrusion_topics_score)/len(self.intrusion_topics_score)
        return self.intrusion_score

    def ctc_rating(self):
        self.rating_topics_score = []
        for topic in self.topics:
            topic_words = ", ".join(topic)
            prompt = "I have a topic that is described by the following keywords: [" + topic_words + "]. Evaluate the interpretability of the topic words on a 3-point scale where 3=“meaningful and highly coherent”  and 0=“useless” as topic words are usable to search and retrieve documents about a single particular subject in the following format: score: <score> without explanation."
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=[{"role": "user", "content": prompt}])
            answer = response['choices'][0]['message']['content']
            answer = answer.replace('\n', ', ')
            parts = answer.split(':')
            score = int(parts[1].split(',')[0])
            self.rating_topics_score.append(score)
        self.rating_score =sum(self.rating_topics_score)/len(self.rating_topics_score)
        return self.rating_score