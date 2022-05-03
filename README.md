# Abstract

There is an increasing trend in using neural methods for dialogue model evaluation. Lack of a framework to investigate these metrics can cause dialogue models to reflect their biases and cause unforeseen problems during interactions. In this work, we propose an adversarial test-suite which generates problematic variations of various dialogue aspects, _e.g._ logical entailment, using automatic heuristics. We show that dialogue metrics for both open-domain and task-oriented settings are biased in their assessments of different conversation behaviors and fail to properly penalize problematic conversations, by analyzing their assessments of these problematic examples. We conclude that variability in training methodologies and data-induced biases are some of the main causes of these problems. We also conduct an investigation into the metric behaviors using a black-box interpretability model which corroborates our findings and provides evidence that metrics pay attention to the problematic conversational constructs signaling a misunderstanding of different conversation semantics.

# URL to the paper

To be added.

# Repository Structure

### Adversarial Conversation Generator
The code to our mutation generator is in the directory _adversary_generator_. 
The code for generating adversarial conversations resides in "generate_mutations.py". Please run the following to obtain the available runtime options:  

python generate_mutations.py --help  

The adversarial generator expects a conversation sampler. An example implementation is in the "sample.py" file.  

### Score Comparison
Some utilities to run a metric on both adversarial and ground-truth conversations is present in the "compare_conversations.py" file. It expects a file containing implementaion of a conversation_scorer class which is an abstraction representing a metric under investigation. An example can be found in "repo_root/DialogRPT/score.py".  

python compare_conversations.py --help  
This can be run to look at the available options for running the script.


# Python Version

This library is tested using 3.8.10 version of Python.
It is advised to use a virtual environment in testing the library out. _requirements.txt_ lists the installed python packages used for testing this test-suite.

# AdversarialEvaluator-AmazonInternship

These are the library commits we used in testing of this test-suite.
Commit Hash for NeuralCoref library: 457c10d7c9828967fbbc9b2d68284c80c78a9adb  
Commit Hash for spacy-lookups-data: 24da7d9c42432cefe26261a2b2ebc8ba259ed915  
Commit Hash for SHAP library version I used: d0b4d59f96adc5d067586c0dd4f7f2326532c47a  
