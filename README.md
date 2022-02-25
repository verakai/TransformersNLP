##### Using a BERT (Bidirectional Encoder Representations from Transformers) base model

---

The goal of this project was to build a a sentiment classifier trained to predict movie reviews. I used Rotten Tomatoe's movie reviews dataset found [here.](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)

Most of the coding was done on PyCharm, however, training was done in Google Colab thanks to their faster GPU.

I used the BERT (Bidirectional Encoder Representations from Transformers) tokenizer for my dataset, and later a pre-trained BERT (bert-base-cased) base model. More about the pre-trained model I used can be found [here.](https://huggingface.co/bert-base-cased)

After loading the data I needed to prepare it to create two input tensors: input ID's and Attention Mask. These were needed to complement the BERT base model used. [Here](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) you can read an interesting explanation about BERT and how to use it. [TensorFlow Hub](https://tfhub.dev/google/collections/experts/bert/1) has multiple versions of BERT base pre-trained models you can use too.