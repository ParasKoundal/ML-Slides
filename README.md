# From Decision Trees to LLMs

A five-lecture series on how machine-learning models learn, starting with decision trees
and ending with large language models. The focus is on **how training actually works**:
what a model is, how its mistakes are measured, and how it improves. No machine-learning
background is assumed.

Every lecture answers the same three questions:

1. **Model**: which family of functions are we choosing from?
2. **Loss**: how do we measure how wrong a choice is?
3. **Learning**: how do we move to a better choice?

---

## Lecture 1 — What does it mean for a machine to learn?
*Decision trees, random forests, gradient boosting and XGBoost*

You will learn:
- The core vocabulary of machine learning: features, labels, parameters, hyper-parameters and loss
- Why models are judged on new data, using training, validation and test sets
- How a decision tree makes predictions, and how it is **built step by step** from data, using impurity (Gini, entropy) and information gain
- Why deep trees **overfit**, what bias and variance mean
- How **random forests** combine many trees through bootstrap sampling and random feature subsets, and why averaging reduces error
- How to read feature importance
- How **gradient boosting** builds a model from small trees, each fixing the errors left by the ones before
- Why boosting is really gradient descent on predictions, and what the learning rate and early stopping do
- How **XGBoost** improves boosting with curvature information

## Lecture 2 — Learning by gradients
*From linear regression to neural networks*

You will learn:
- Linear and logistic regression, and where common loss functions come from (squared error, cross-entropy)
- **Gradient descent**: loss landscapes, the learning rate, and stochastic mini-batches
- How stacking simple units with nonlinearities gives a neural network, and why the nonlinearity matters
- **Backpropagation** as the chain rule applied through a network, worked by hand on a small example
- What happens during training, step by step, as a small network learns a decision boundary

## Lecture 3 — Making deep learning work
*Training practice and architectures*

You will learn:
- Modern optimisers (momentum, Adam), and why initialisation matters
- Vanishing and exploding gradients, and how normalisation and residual connections fix them
- Regularisation: weight decay, dropout and early stopping
- Architectures as built-in assumptions about data: CNNs for images, RNNs for sequences, and their limits
- **Embeddings**: how words and other discrete things become vectors whose geometry carries meaning

## Lecture 4 — Attention and the Transformer

You will learn:
- **Tokenization**: how text becomes numbers a model can process
- **Self-attention** built up step by step: queries, keys, values and attention weights
- How a full Transformer block is assembled: multi-head attention, feed-forward layers, residuals, normalisation and positional information
- The training objective: **next-token prediction**, and why the model cannot see the future
- That this is trained with the same loss and the same backpropagation as Lecture 2

## Lecture 5 — From a Transformer to an LLM
*Scale, alignment and behaviour*

You will learn:
- **Pre-training at scale**, and the scaling laws that link model size, data and compute to performance
- How text is generated: sampling, temperature, and why outputs vary
- How a text predictor becomes an assistant: supervised fine-tuning and preference training (RLHF, DPO)
- Emergent abilities such as in-context learning
- Known limitations such as hallucination, and why they follow from how the model is trained

---

## How the lectures are taught

- Concepts are built from the ground up, with worked examples done by hand before the algorithm runs
- Animations computed from the real algorithms show learning as it happens
- Examples from bioprocess and chemical engineering, on clearly labelled simulated data

## Author

**Paras Koundal**
