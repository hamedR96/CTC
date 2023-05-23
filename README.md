# CTC

## Python Package
CTC is implemented as a service for researchers and engineers who aim to evaluate and fine-tune their topic models. The source code of this python package is provided in `./ctc` and a notebook named `example.ipynb` is prepared to explain how to use this python package as follows.

### Automated CTC
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
```python
from ctc.main import Semi_auto_CTC

openai_key="YOUR OPENAI KEY"

y=Semi_auto_CTC(openai_key,topics)

y.ctc_intrusion()

y.ctc_rating()
```

## Experiments
The experiments of this paper including topic models, measuring scores, and complete results are provided in the `./experiment` folder. To reproduce the results from scratch, it is required first to train topic models with the notebooks provided in `./experiment/NTM`. Afterward, there are 4 notebooks in `./experiment` to compute CTC<sub>CPMI</sub>, CTC<sub>Intrusion</sub>, CTC<sub>Rating</sub>, and traditional topic coherence metrics and to save the results as `.txt` files. The analysis is provided in the notebook named `Analysis.ipynb`.
