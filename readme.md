# VicunaABSA - Instruction learning with Vicuna for Aspect-Based Sentiment Analysis tasks

This repository contains all the tools used in my master's thesis *Prompt Engineering & In-Context Learning: Text Generation Transformers for Aspect-Based Sentiment Analysis*.

`requirements.txt` contains the Python package requirements needed to run the code in this repository.

## Datasets

`datasets/` contains custom loading scripts as well as raw data, for the SemEval 2014 Task 4, Semeval 2015 Task 12 and SemEval 2016 Task 5 datasets, which were originally provided in XML format.

All these datasets were explored during the project, but the scope of the experiment was reduced to the restaurants 2015 and 2016 datasets since these had the most complete set of annotations.

We tried to preserve as much as possible the original tree structure of some of the data, in order to explore the original SemEval tasks as they were defined. This was not leveraged in the final experiment setting, but it was explored earlier in the project.

The data was made available to the public through the HuggingFace Hub and can be previewed here:
- https://huggingface.co/datasets/alexcadillon/SemEval2014Task4
- https://huggingface.co/datasets/alexcadillon/SemEval2015Task12
- https://huggingface.co/datasets/alexcadillon/SemEval2016Task5

 ## Data inference tools

`data_inference_tools.py` contains all the tools specifically built to run experiments on the available datasets.

`DatasetInferenceHandler` classes handle the inference pipeline from start to end: prompt generation, Vicuna calls through the HuggingFace Inference API, and output post-processing. Metrics are also defined.

The `DatasetAnalyzer` class handles asynchronous inference and computes metrics on a dataset level.

For ease of use, `prompt_structures.json` and `validation_schemas.json` were created to easily change the pipeline properties that are dataset dependent.

## Parameter tuning

`parameter_tuning.ipynb` contains the process that was followed to select inference parameters, using the data inference tools.

The results and the figures presented in the thesis appendix are stored in `parameter_tuning/`.

## Model testing

The experiments presented in the *Results* section in the thesis are run in `model_testing.ipynb`, using the data inference tools.

The results are stored in `model_testing/`, with some additional metadata, to be able to identifying each experiment's settings.