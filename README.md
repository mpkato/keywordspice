Keywordspice
============

This is a Python implementation of "Oyama, Kokubo, and Ishida: Domain-Specific
Web Search with Keyword Spices, TKDE 2004".

Requirements
-----------
- Numpy

Preparation
-----------
Please first prepare **labeled data** 
and separate it into **training data** and **validation data**.

**Labeled data** is a set of documents with a positive/negative label.
For example, you can download a hundred of documents, 
and label them as either recipe-related or not when you want to develop
a recipe search engine.

**Training data** is a part of **labeled data** and used for training a
decision tree, 
while **validation data** is the rest of it and used 
for refining the trained tree. See (Oayama+, TKDE2004) for the details.

Both of the data must be stored in different files,
and each line in the files must be of the following format:

`<ID> <Label> <Document>`

where &lt;ID&gt; is a unique identifier in the labeled data,
&lt;Label&gt; is either 1 (positive) or 0 (negative),
and &lt;Document&gt; is a list of words separated by whitespaces.
Note that variables should be separated by **TAB**,
and &lt;Document&gt; should include only pre-processed words
(e.g. stopwords are excluded).

Usage
---------
Run keywordspice.py after preparing trainding and validation data.
The usage of keywordspice.py is shown below:

`keywordspice.py train_filepath valid_filepath`

where *train_filepath* is a filepath for training data,
while *valid_filepath* is a filepath for validation data.

Then, you will find a keyword spice in the stdout.

