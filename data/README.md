# Questions
## US2016
`quest_us2016.csv`
* `argid`: Argument ID.
* `locid`: Locution ID.
* `loctext`: Locution text.
* `propid`: Proposition ID.
* `proptype`: Proposition type (from the original corpus).
* `ya`:  Illocutionary act.
* `proptext`: Proposition text.
* `proptext_processed`: Proposition text processed by CoreNLP.
* `loctext_processed`: Locution text processed by CoreNLP.

## MoralMaze
`quest_mm2012.csv`
* `argid`: Argument ID.
* `locid`: Locution ID.
* `loctext`: Locution text.
* `propid`: Proposition ID.
* `proptype`: Proposition type (from the original corpus).
* `ya`:  Illocutionary act.
* `proptext`: Proposition text.
* `proptext_processed`: Proposition text processed by CoreNLP.
* `loctext_processed`: Locution text processed by CoreNLP.


# Reported Speech
## PARC 3.0
`rspch_parc.csv`
* `propid`: Proposition ID.
* `split`: Split (train, val, test).
* `text`: Text.
* `source`: Speech source.
* `source_mask`: Word-level BIO tags for the speech source.
* `content`: Speech content.
* `content_mask`: Word-level BIO tags for the speech content.
* `text_nlp`: Text processed by CoreNLP.

## US2016
`rspch_us2016.csv`
* `fold`: Fold (0--4).
* `propid`: Proposition ID.
* `argid`: Argument ID.
* `corpus`: Original corpus name.
* `speaker`: Speaker.
* `text`: Text.
* `is_source`: 1 if this instance is reported speech; 0 otherwise.
* `source`: Speech source.
* `content`: Speech content.
* `credibility`: Credibility of the speaker.
* `text_nlp`: Text processed by CoreNLP.
