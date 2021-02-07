# Multilingual plagiarism detection

> As a diploma project for the bachelor's degree at `National University of Kyiv-Mohyla Academy (NaUKMA)`

## Author

Danylo Kravchenko, applied mathematics 4 <br/>
Email: <a href="emailto:kravchel16@gmail.com">kravchel16@gmail.com</a> <br/>
Diploma official name: `Applying Deep Learning for text analysis`

## The Project

The system allows detecting plagiarism in different languages. The currently available options are Ukrainian and English. 
The system takes 2 texts in any language from available ones and detects plagiarism in them.

The heart of the system is a Keras model built for the binary classification of texts. The base architecture of the model is a `BERT` transformer.
Unfortunately, there are a few plagiarism datasets on the internet and they are quite small for training, so I've taken the pre-trained `bert_multi_cased_L-12_H-768_A-12` model that was originally trained on Wikipedia and a book corpus. Then, I've fine-tuned it on the [SNLI courpus](https://nlp.stanford.edu/projects/snli/) to make the model classify the similarity of the texts. The final step is to fine-tune the model on the [plagiarism dataset](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html).

The whole research is located in the `Jupyter` notebook in the `research` directory.

`Rust` web server loads the saved Keras model from a disk using special lib `PyO3` that allows using `Python` in the `Rust` application. The web server handles users' requests and detects plagiarism in the given texts.

## Translations

The original dataset is in English and my task was to make the model bilingual. That's why I've translated the dataset to Ukrainian using `Google Cloud Translation`.

You may see the code in the `translation` directory.

## Results

| Text A        | Text B           | Result  |
| ------------- |:-------------:| -----:|
| Functional programming (often abbreviated FP) is the process of building software by composing pure functions, avoiding shared state, mutable data, and side-effects. Functional programming is declarative rather than imperative, and application state flows through pure functions. Contrast with object oriented programming, where application state is usually shared and colocated with methods in objects. Functional programming is a programming paradigm, meaning that it is a way of thinking about software construction based on some fundamental, defining principles (listed above).      | Functional programming is a programming paradigm, that allows to develop software by composing functions, keeping away from shared state, mutable data and side-effects. In functional programming application state goes through pure functions. This paradigm is opposite to applications that were built using the object oriented programming. State in those applications is often shared and compiled with methods in objects. Functional programming is rather a way of thinking about developing software based on fundamental rules. | plagiarism - `99.99%` |
| In mathematics and statistics, the arithmetic mean, or simply the mean or the average (when the context is clear), is the sum of a collection of numbers divided by the count of numbers in the collection.      | Арифметичне середнє (в математиці і статистиці) — сума всіх фіксованих значень набору, поділена на кількість елементів набору      |   plagiarism - `97.67%` |
| In object-oriented programming, inheritance is a way to form new classes (instances of which are called objects) using classes that have already been defined. The inheritance concept was invented in 1967 for Simula. The new classes, known as derived classes, take over (or inherit) attributes and behavior of the pre-existing classes, which are referred to as base classes (or ancestor classes). It is intended to help reuse existing code with little or no modification. Inheritance provides the support for representation by categorization in computer languages. Categorization is a powerful mechanism number of information processing, crucial to human learning by means of generalization (what is known about specific entities is applied to a wider group given a belongs relation can be established) and cognitive economy (less information needs to be stored about each specific entity, only its particularities).Inheritance is also sometimes called generalization, because the is-a relationships represent a hierarchy between classes of objects. For instance, a "fruit" is a generalization of "apple", "orange", "mango" and many others. One can consider fruit to be an abstraction of apple, orange, etc. Conversely, since apples are fruit (i.e., an apple is-a fruit), apples may naturally inherit all the properties common to all fruit, such as being a fleshy container for the seed of a plant. An advantage of inheritance is that modules with sufficiently similar interfaces can share a lot of code, reducing the complexity of the program. Inheritance therefore has another view, a dual, called polymorphism, which describes many pieces of code being controlled by shared control code. Inheritance is typically accomplished either by overriding (replacing) one or more methods exposed by ancestor, or by adding new methods to those exposed by an ancestor. Complex inheritance, or inheritance used within a design that is not sufficiently mature, may lead to the Yo-yo problem. | In object oriented programming, objects are grouped together into classes according to their type, structure and the functions that can be performed on them. Inheritance is a process in object oriented programming in which objects acquire (or inherit) the properties of objects of another class. It is therefore used to create relationships between one object and another. Each class groups together objects of a similar type, with similar properties. New classes can be formed by this process whose objects will have properties of both the classes from which this new class is formed. A superclass has all of the properties of the subclasses below it. At the same time subclasses are each distinctive from each other but related via the superclass. Subclasses are said to ‘extend’ superclasses. Due to these relationships, object oriented programmes tend to be easier to modify since they do not need to be changed when a new object, with different properties is added. Instead, a new object is made to inherit properties of objects which already exist. Inheritance can be divided into two main processes: single inheritance and multiple inheritance. Single inheritance means that the class can only inherit from one other class, whereas multiple inheritance allows for inheritance from several classes.      |    non plagiarism - `0.85%` |
| A manager is a person who is responsible for a part of a company, i.e., they ‘manage‘ the company. Managers may be in charge of a department and the people who work in it. In some cases, the manager is in charge of the whole business.  The Manager’s duties also include managing employees or a section of the company on a day-to-day basis. For example, a restaurant manager is in charge of the whole restaurant. | A manager is a person responsible for controlling or administering an organization or group of staff. Managers could be responsible for a part of the company and the employers who work there. Sometimes the manager is a head of the whole company. Reponsibilities of the manager include control workers or lead a part of the company too. For instance, a store manager is responsible for the whole store. | plagiarism - `99.08%` |
| Тара́с Григо́рович Шевче́нко (відомий також як Кобза́р; 25 лютого (9 березня) 1814, с. Моринці, Київська губернія, Російська імперія (нині Звенигородський район, Черкаська область, Україна) — 26 лютого (10 березня) 1861, Санкт-Петербург, Російська імперія) — український письменник, класик української літератури, мислитель, художник. Національний герой і символ України. | Franklin Delano Roosevelt (/ˈroʊzəvəlt/,[1] /-vɛlt/[2] ROH-zə-velt; January 30, 1882 – April 12, 1945), often referred to by his initials FDR, was an American politician who was the 32nd president of the United States from 1933 until his death in 1945. A member of the Democratic Party, he won a record four presidential elections and became a central figure in world events during the first half of the 20th century.  | non plagiarism - `0.03%` |

## Accuracy

The accuracy of the model is really impressive. I've achieved 100% accuracy on the `train`, `valid`, and `test` datasets. See the results above for examples.

## How to setup `PyO3`

I suggest using `miniconda` or `anaconda` as a virtual environment for your `Python` since it has a clear structure of directories and it is easy to find the interpreter and loaded python modules.

You need to set environment variables `LD_LIBRARY_PATH` and `PYO3_PYTHON` to link `Python` and `PyO3`. For example, using `miniconda` I've set them in this way:
```sh
export LD_LIBRARY_PATH='/home/user/miniconda3/lib/'
export PYO3_PYTHON='/home/user/miniconda3/bin/python3'
```
The requirements for the `Python` code are in the `requirements.txt` file.
