[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://hamedrahimi.fr)
[![arXiv](https://img.shields.io/badge/arXiv-2302.14587-<COLOR>.svg)](https://arxiv.org/abs/2305.14587)


# CTC: Contextualized Topic Coherence Metrics

This research introduces a new family of topic coherence metrics called Contextualized Topic Coherence Metrics (CTC) that benefits from the recent development of Large Language Models (LLM). CTC includes two approaches that are motivated to offer flexibility and accuracy in evaluating neural topic models under different circumstances. Our results show automated CTC outperforms the baseline metrics on large-scale datasets while semi-automated CTC outperforms the baseline metrics on short-text datasets.

CTC is implemented as a service for researchers and engineers who aim to evaluate and fine-tune their topic models. The source code of this python package is provided in `./ctc` and a notebook named `example.ipynb` is prepared to explain how to use this python package as follows.

### Automated CTC
We adopt CPMI for introducing a new automated Contextualized Topic Coherence (CTC) metric. The following figure illustrates the computation of automated CTC, which estimates statistical dependence within a topic in a corpus by calculating CPMI between every pair of topic words within a sliding window. Therefore, the first step in this procedure is to split the corpus into a set of window segments with a length of $w$ that have $k$ words intersection with adjacent window segments. Afterward, we compute the CPMI between each pair of words within each topic, and average over all the window segments

![alt text](https://github.com/hamedR96/CTC/blob/main/formula.png?raw=true)
![alt text](https://github.com/hamedR96/CTC/blob/main/cpmi.jpg?raw=true)

with the following code you can calculate $CTC_{CPMI}$:
```python
from ctc.main import Auto_CTC
#initiating the metric
eval=Auto_CTC(segments_length=15, min_segment_length=5, segment_step=10,device="mps") 

# segmenting the documents
docs=documents 
eval.segmenting_documents(docs) 

# creating cpmi tree including all co-occurence values between all pairs of words 
eval.create_cpmi_tree() 
#eval.load_cpmi_tree() 

# topics=[["game","play"],["man","devil"]] for instance
eval.ctc_cpmi(topics) 
```

### Semi-automated CTC


#### Intrusion
To assess the quality of topic models, we measure a coherence score that takes into account a low probability for intruder words to belong to a topic. We adopt this notion to chatbots with the following prompt, which provides the topic words to ChatGPT and asks for a category and intruder words.


>I have a topic that is described by the following keywords:[topic_words].
       Provide a one-word topic based on this list of words and identify all 
    intruder words in the list with respect to the topic you provided. Results be 
      in the following format: topic: "one-word", intruders: "words in a list"

#### Rating

We adapt this idea to chatbots with the following prompt, which provides the topic words to ChatGPT and asks to rate the usefulness of the topic words for retrieving documents on a given topic. The $CTC_{\text{Rating}}$ for a topic model is then obtained by the average sum of all ratings over all the topics. 

>I have a topic that is described by the following keywords: [topic_words]. 
      Evaluate the interpretability of the topic words on a 3-point scale where
       3=“meaningful and highly coherent”  and 0=“useless” as topic words are 
      usable to search and retrieve documents about a single particular subject. 
      Results be in the following format: score: "score"

```python
from ctc.main import Semi_auto_CTC

openai_key="YOUR OPENAI KEY"

y=Semi_auto_CTC(openai_key,topics)

y.ctc_intrusion()

y.ctc_rating()
```

## Citation
To cite [CTC](https://arxiv.org/abs/2305.14587), please use the following bibtex reference:
```bibtext
@misc{rahimi2023contextualized,
      title={Contextualized Topic Coherence Metrics}, 
      author={Hamed Rahimi and Jacob Louis Hoover and David Mimno 
      and Hubert Naacke and Camelia Constantin and Bernd Amann},
      year={2023},
      eprint={2305.14587},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
