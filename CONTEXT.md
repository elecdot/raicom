# RAICOM Weather Classification

This context names the shared language for the RAICOM weather image classification competition workspace. It keeps contest concepts distinct from local implementation choices.

## Language

**Weather Image Classification**:
A four-class image classification task where each input image is assigned one weather category.
_Avoid_: Generic image classification, weather detection

**Weather Category**:
One of the official output labels: `sunny`, `cloudy`, `rainy`, or `snowy`.
_Avoid_: Class name, label string, weather type

**Training Set**:
The labeled images provided for model development before submission.
_Avoid_: Dataset, train data, local images

**Training Image Root**:
The Training Set directory that directly contains the four Weather Category subdirectories.
_Avoid_: Dataset root, archive root, train/train

**Internal Validation Split**:
A held-out portion of the Training Set used locally to estimate generalization before submitting.
_Avoid_: Test set, competition set, public score

**Competition Set**:
The hidden image set used by the competition platform for final scoring.
_Avoid_: Validation set, local test set, train split

**Macro F1**:
The official ranking metric that gives each Weather Category equal weight regardless of class frequency.
_Avoid_: Accuracy, weighted F1, micro F1

**Submission Prediction Interface**:
The platform-facing prediction entry point that receives one image and returns one Weather Category.
_Avoid_: Training script, notebook cell, evaluation helper

**Model Artifact**:
A trained model file produced from the Training Set and later loaded by the Submission Prediction Interface.
_Avoid_: Result file, checkpoint, output blob

**Platform Run**:
The execution of the submitted project on the hosted momodel environment for system testing or scoring.
_Avoid_: Local run, offline experiment, notebook test
