# Bilingual plagiarism detection

> As a diploma project for the bachelor's degree at `National University of Kyiv-Mohyla Academy (NaUKMA)`

## Author

Danylo Kravchenko, applied mathematics 4 <br/>
Email: <a href="emailto:kravchel16@gmail.com">kravchel16@gmail.com</a> <br/>
Diploma official name: `Applying Deep Learning for text analysis`

## The Project

The system allows to detect plagiarism in Ukrainian and English texts. 

- English to English
- Ukranian to Ukranian
- Ukrainian to English
- English to Ukranian

The heart of the system is a Keras model built for the binary classification of texts. The base architecture of the model is a `BERT` transformer.
Unfortunately, there are a few plagiarism datasets on the internet and they are quite small for training, so I've taken the pre-trained `bert_multi_cased_L-12_H-768_A-12` model that was originally trained on Wikipedia and a book corpus. Then, I've fine-tuned it on the [SNLI courpus](https://nlp.stanford.edu/projects/snli/) to make the model classify the similarity of the texts. The final step is to fine-tune the model on the plagiarism dataset.

The whole research is located in the `Jupyter` notebook in the `research` directory.

`Rust` web server loads the saved Keras model from a disk using special lib `PyO3` that allows using `Python` in the `Rust` application. The web server handles users' requests and detects plagiarism in the given texts.

## Translations

The original dataset is in English and my task was to make the model bilingual. That's why I've translated the dataset to Ukrainian using `Google Cloud Translation`.

You may see the code in the `translation` directory.

## Accuracy

The accuracy of the model is really impressive. I've achieved 100% accuracy on the `train`, `valid`, and `test` datasets.

## How to setup `PyO3`

I suggest using `miniconda` or `anaconda` as a virtual environment for your `Python` since it has a clear structure of directories and it is easy to find the interpreter and loaded python modules.

You need to set environment variables `LD_LIBRARY_PATH` and `PYO3_PYTHON` to link `Python` and `PyO3`. For example, using `miniconda` I've set them in this way:
```sh
export LD_LIBRARY_PATH='/home/user/miniconda3/lib/'
export PYO3_PYTHON='/home/user/miniconda3/bin/python3'
```