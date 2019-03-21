# DeViSE: A Deep Visual-Semantic Embedding Model
[Fromme et al. (2013)](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)

[Try the model on AWS](http://ec2-18-195-116-70.eu-central-1.compute.amazonaws.com:8000/apidocs/)!

DeViSE maps images into a rich semantic embedding space to combine the image recognition task with semantic information about the similarity of words/categories learned from language models. The method can make predictions about thousands of image labels not observed during training (zero-shot predictions).

The training notebook is based on [fast.ai (part 2, lesson 11)](https://youtu.be/tY0n9OT5_nA?t=6930) but uses a custom PyTorch model. The model is deployed to AWS using Flask, Swagger UI, and Docker.

## Getting Started
### Prerequisites
Create a conda environment to run the notebook and train the model or to run the app using the flask development server.

Install anaconda and then run:

```
conda install nb_conda_kernels
conda env create -f env.yml
ipython kernel install --user --name=DeViSE_env
source activate DeViSE_env
```

Then install [fastText](https://github.com/facebookresearch/fastText/tree/master/python):

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
```

### Train the model
A model fine-tuned on ImageNet is provided in this repository. If you want to train the model yourself, follow these steps:

Download the needed datasets:
1. [Full ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge)
2. Or a subset of [ImageNet](http://files.fast.ai/data/imagenet-sample-train.tar.gz)
3. [FastText word vectors](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip)

Run the notebook `DeViSE - A Deep Visual-Semantic Embedding Model.ipynb`

After running the notebook, you can replace the model I provided with the one you trained by copying the files `devise_trained_full_imagenet.pth` and `className_2_wordvec_without_dups.pkl` to the folder `deviseApi`.

## Running the flask development server
Put a folder/several folders with a selection of pictures, i.e. from the ImageNet validation dataset, in a folder called `pictures/` in the folder `deviseApi`.

The folder structure should look for example like this:

```
|-- deviseApi
|   |-- pictures
|   |   |-- <folder_category_1>
|   |   |   |-- <picture1>.JPEG
|   |   |   |-- <picture2>.JPEG
|   |   |-- <folder_category_2>
|   |   |   |-- <picture1>.JPEG
|   |   |   |-- <picture2>.JPEG
.
.
.
```

or like this:

```
|-- deviseApi
|   |-- pictures
|   |   |-- <folder_all_categories_mixed>
|   |   |   |-- <picture1>.JPEG
|   |   |   |-- <picture2>.JPEG
|   |   |   |-- <picture3>.JPEG
|   |   |   |-- <picture4>.JPEG
.
.
.
```
Folder and image names within the folder `pictures` do not matter.

Activate the virtual environment with `source activate DeViSE_env`, navigate to the folder `deviseApi` and run `python deviseApi.py`. In your browser go to `http://localhost:5000/apidocs/`.

## Dockerfile: Build and run
Install docker. Navigate to the parent folder and run:

```
docker build -t devise .
docker run -d -p 8000:8000 devise
```

In your browser go to `http://localhost:8000/apidocs/`.

Stop the running container:

Run `docker ps` to get the CONTAINER ID. Then run `docker stop <CONTAINER ID>`

