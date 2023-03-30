# CAKE: Classifiers Are Keyphrase Explainable
Cake is a method of local interpret-ability for Transformer classifiers that deal with text. It is based on keyphrase extraction pipeline and can handle single label and multi label data. CAKE is extended with a surrogate model to CAKES in order to be model agnostic.
a preliminary version of the paper introducing CAKE can be found [link]

## Datasets included
both BERT models and DistilBERT models finetuned on each one can be found [here](https://drive.google.com/drive/folders/1fkNhyVUhalxn3yM6rUCJ5lBD7T88yH6-?usp=sharing) from the optimus paper.
- HateXplain
- Hummingbird
- HOC
- AIS
- Movies
- Ethos
- ESNLI


## Disclaimer
the code provided here is made to be completely compatible with [Optimus](https://github.com/intelligence-csd-auth-gr/Optimus-Transformers-Interpretability), code taken from their repository is inside ``optimus_code`` folder and contains helper functions, model loaders, and dataset loaders in order to be identical and directly comparable. Additionaly the same finetuned models were used.
This also means the main evaluation loop as well as the code structure presented in ``masterscript.py`` follows a similar generalized pattern with a different explainer.

# Running expirements
This process is the same for any expirement, with either CAKE or CAKES and any model BERT or DistilBERT.
**Installation:**
-	clone the repository
- ``$ pip install -r requirements.txt``
- replace _model.py in keybert library (ex for **venv**: venv\Lib\site-packages\keybert\)
- extract models donwloaded from [here](https://drive.google.com/drive/folders/1fkNhyVUhalxn3yM6rUCJ5lBD7T88yH6-?usp=sharing) in the model folder.
> name each model {bert or distilbert}_{dataset identifier in lowercase} ex. distilbert_hx
> identifiers include ais, esnli, ethos, hoc, hummingbird, hx, movies.


**Running expirements:**
- run ``combinations.py`` to generate the parameters for the expirements you want to run
- make sure ``parameters.xlsx`` contain all the combinations you want to run
- run ``masterscript.py`` to fill the blank results in ``parameters.xlsx`` *(will need multiple runs)*
>	optional: run ``splitresults.py`` in order to split the results in different files per group of parameters for easier interpretation.
 
# Generating Parameter combinations
the generation script ``combinations.py`` consist of simple arrays containing 0 or 1 for each parameter. parameters that have 1, will generate every possible combination with other parameters that have 1 and output ``parameters.xlsx``. parameters that have 0 will instead generate only a single variation of the default parameters and output in ``combo.xlsx``. From those you can pick which ones should run and append them to ``parameters.xslx`` generation are set to overwrite previous generations so if you added results to the empty columns of ``parameters.xlsx`` they would be deleted.

# Frequently asked questions
## generation script taking forever or crash out of memory
since you are generating every possible combination if you chose too many parameters to check, the number of combinations grows exponentially and can result in out of memory computing the combinations. chose a smaller number of parameters to check since that many won't be feasible to actually run.

## masterscript didn't fill all the blank lines in parameters

masterscript will fill the blanks for only one group of parameters each time. 
groups are defined by (CAKE or CAKES, BERT or DistilBERT, Dataset)
in our expirements there are 20 groups total.

## HateXplain with DistilBERT throws errors

 The HateXplain DistilBERT model while being similar to the other 6 Distiled models currently is not working when trying to load it with the flair library.
 You can either delete the rows for this combination of parameters in parameters.xlsx or skip the selected rows









