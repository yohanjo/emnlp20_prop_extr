# Questions
You may start from `run_quest_trans.sh`. Modify the parameters as you need and run it:
```
$ run_quest_trans.sh [MODE]
```

`[MODE]`
* `baseline`: No transformation.
* `basic`: Basic model.
* `copy`: Copy model.
* `rule`: Rule-based model

The neural models use GloVe embeddings. You should download `glove.840B.300d.txt` ([link](http://nlp.stanford.edu/data/glove.840B.300d.zip)) to the `../rsc` folder.

Required libraries: `pytorch`, `tqdm`


# Reported Speech
## PARC3.0
### BERT
```
$ python clf_rspch_bert_parc3.py -mode [MODE]
```

`[MODE]`
* `content`: Content identification.
* `source`: Source identification.

Required libraries: `pytorch`, `pytorch_transformers`, `tqdm`

### CRF
```
$ python clf_rspch_crf_parc3.py -mode [MODE]
```

`[MODE]`
* `content`: Content identification.
* `source`: Source identification.

Required libraries: `sklearn_crfsuite`


## US2016
### BERT
Content identification
```
$ python clf_rspch_content_bert_us2016.py 
```

Source identification
```
$ python clf_rspch_source_bert_us2016.py 
```

Required libraries: `pytorch`, `pytorch_transformers`, `tqdm`

### CRF
```
$ python clf_rspch_crf_us2016.py -mode [MODE]
```

`[MODE]`
* `content`: Content identification.
* `source`: Source identification.

Required libraries: `sklearn_crfsuite`


