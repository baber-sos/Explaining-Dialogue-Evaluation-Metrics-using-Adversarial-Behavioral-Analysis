# Abstract

There is an increasing trend in using neural methods for dialogue model evaluation. Lack of a framework to investigate these metrics can cause dialogue models to reflect their biases and cause unforeseen problems during interactions. In this work, we propose an adversarial test-suite which generates problematic variations of various dialogue aspects, _e.g._ logical entailment, using automatic heuristics. We show that dialogue metrics for both open-domain and task-oriented settings are biased in their assessments of different conversation behaviors and fail to properly penalize problematic conversations, by analyzing their assessments of these problematic examples. We conclude that variability in training methodologies and data-induced biases are some of the main causes of these problems. We also conduct an investigation into the metric behaviors using a black-box interpretability model which corroborates our findings and provides evidence that metrics pay attention to the problematic conversational constructs signaling a misunderstanding of different conversation semantics.

# URL to the paper

The URL to our paper can be found at this link.

# Repository Structure

# Python Version

This library is tested using 3.8.10 version of Python.
It is advised to use a virtual environment in testing the library out. _requirements.txt_ lists the installed python packages used for testing this test-suite.

# AdversarialEvaluator-AmazonInternship

These are the library commits we used in testing of this test-suite.
Commit Hash for NeuralCoref library: 457c10d7c9828967fbbc9b2d68284c80c78a9adb  
Commit Hash for spacy-lookups-data: 24da7d9c42432cefe26261a2b2ebc8ba259ed915  
Commit Hash for SHAP library version I used: d0b4d59f96adc5d067586c0dd4f7f2326532c47a  
