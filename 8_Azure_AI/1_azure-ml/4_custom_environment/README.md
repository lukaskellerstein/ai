Create a custom environment via AML UI

- it is just a docker image with a custom dependencies

Base Docker images (cpu and gpu):
https://github.com/Azure/AzureML-Containers/blob/master/README.md

YAML config with custom dependencies and python libraries

```
name: pytorch-env
channels:
  - defaults
  - pytorch
dependencies:
  - python=3.10
  - pip=23.0.1
  - pip:
      - numpy==1.24.3
      - pandas==2.0.3
      - gensim==4.3.1
      - torch==2.0.1
      - scikit-learn==1.3.0
      - nltk==3.8.1
```
