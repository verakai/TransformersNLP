##### Using a BERT (Bidirectional Encoder Representations from Transformers) base model

---

The goal of this project was to build a a sentiment classifier trained to predict movie reviews. I used Rotten Tomatoe's movie reviews dataset found [here.](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)

Most of the coding was done on PyCharm, however, training was done in Google Colab thanks to their faster GPU.

I used the BERT (Bidirectional Encoder Representations from Transformers) tokenizer for my dataset, and later a pre-trained BERT (bert-base-cased) base model. More about the pre-trained model I used can be found [here.](https://huggingface.co/bert-base-cased)

After loading the data I needed to prepare it to create two input tensors: input ID's and Attention Mask. These were needed to complement the BERT base model used. [Here](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) you can read an interesting explanation about BERT and how to use it. [TensorFlow Hub](https://tfhub.dev/google/collections/experts/bert/1) has multiple versions of BERT base pre-trained models you can use too.

The sentiment labels in the Rotten Tomatoes dataset are the following:

- 0 - negative
- 1 - somewhat negative
- 2 - neutral
- 3 - somewhat positive
- 4 - positive

Additionally the distribution in the dataset was like this:

<p align="center">
    <img src='https://user-images.githubusercontent.com/35600758/155804533-d4248d89-a1ef-4368-b0cd-2344d6d88142.png'
width="400px">
</p>

The model I built and later used for training was the following:

``` {.python}
# Two inputs
input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int64')
mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int64')

# Transformer
embeddings = bert.bert(input_ids, attention_mask=mask)[0]

# Classifier
x = tf.keras.layers.Dropout(0.1)(embeddings)
x = tf.keras.layers.GlobalMaxPool1D()(x)
y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(x)

# Initialize model
model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

# Freeze layer
model.layers[2].trainable = False

model.summary()
```
<p align="center">
    <img src='https://user-images.githubusercontent.com/35600758/155804602-58c10b28-4e5c-40fc-9322-adedf0d98ea3.png'
width="900px">
</p>

Once the model was fitted I saved it to later use it with multiple devices:

``` {.python}
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3)
```
<p align="center">
    <img src='https://user-images.githubusercontent.com/35600758/155804643-7bb7acba-a8cb-4135-9340-8bcca994400b.png'
width="900px">
</p>

``` {.python}
model.save('sentiment_model')
```

As you can see the epochs and batches used are low, yielding in a rather low accuracy. Nevertheless, below you can observe the model in action predicting the different phrases and the result was better than expected:

<div style="padding:60.66% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/682050184?h=e48dd1484d&amp;badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;" title="BERT"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>





##### Access the notebook in my [GitHub repo.](https://github.com/verakai/TransformersNLP)
