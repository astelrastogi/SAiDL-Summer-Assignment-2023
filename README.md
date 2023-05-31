# SAiDL-Summer-Assignment-2023
Repository for submitting solutions to SAiDL Summer Assignment 2023. Implemented Section 1 along with its bonus task, and Section 2 and 3. 


# Section 1

The code is present [here](https://github.com/astelrastogi/SAiDL-Summer-Assignment-2023/blob/main/SAiDL_Section1.ipynb). First use a standard Softmax, and then a Gumbel-Softmax function. 

## Task 1
CNN Model with a standard Softmax activation function: 
```
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(100, activation="softmax"),
    ]
)
```
## Task 2
CNN Model with Gumbel-Softmax that reduces the complexity of softmax function: 
```
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(100, activation=None),
    ]
)

# Define the temperature for the Gumbel-Softmax function
temperature = 0.5
# Compile the model with the custom loss function
model.compile(optimizer="adam", loss=gumbel_softmax_loss, metrics=["accuracy"])

```
The definition and loss calculated using Gumbel Softmax: 
```
def gumbel_softmax(logits, temperature):
    # Sample from a Gumbel distribution
    u = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel = -tf.math.log(-tf.math.log(u + 1e-20) + 1e-20)
    
    # Add the Gumbel noise to the logits and apply temperature
    y = logits + gumbel
    y = y / temperature
    
    # Compute the softmax
    y = tf.nn.softmax(y)
    
    return y
```
```
def gumbel_softmax_loss(y_true, y_pred):
    y_pred = gumbel_softmax(y_pred, temperature)
    loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss
```
## Task 3
Using the standard Softmax function gives an accuracy of around 42%  

<img width="263" alt="Screenshot 2023-04-24 at 11 28 44 AM" src="https://user-images.githubusercontent.com/54110949/233911691-2cfb67c9-42c9-4bf5-bad7-472415261ec2.png">

Whereas using the Gumbel-Softmax, the accuracy comes out to be 42% 

<img width="223" alt="Screenshot 2023-04-24 at 12 02 05 PM" src="https://user-images.githubusercontent.com/54110949/233917130-014a433c-1105-4862-96b1-72e51912ca6d.png">

## Task 4
For standard softmax - 10 epochs of 782 iterations each (took 134s on avg on each epoch) 
For gumbel softmax - 10 epochs of 782 iterations each (took 135s on avg on each epoci\h)

## BONUS

# Section 2A - Natural Language Processing
To complete this task, these steps are required

1. Data Preparation:

   a. Obtain the monolingual English corpus HASOC dataset (Hindi 2019).
   
   b. Preprocessing the data by removing irrelevant characters, punctuation, and normalizing the text.

2. Code Mixing:

   a. Use a standard translator, such as Google Translate or Microsoft Translator API, to create code-mixed sentences. Chose Hindi as a target language for code mixing.
   
   b. Apply code mixing to the English sentences from the dataset by translating them into the target language and then translating them back into English, resulting in code-mixed sentences.
   
   c. Vary the code-mixing index (CMI) to create different levels of code mixing. CMI represents the proportion of words in a sentence that come from the target language.

3. Finetuning the Language Models:

   a. Select pre-trained language models for finetuning. In this case, we will consider both BeRT (base model) and m-BeRT (multilingual model).
   
   b. Preprocess the code-mixed data, including tokenization and encoding, to prepare it for finetuning.
   
   c. Finetune the BeRT and m-BeRT models on the code-mixed dataset. The finetuning process involves training the models on the code-mixed data and updating their weights to improve performance on the task of detecting abuse in social media content.

4. Performance Evaluation:

   a. Use the standard code-mixed HASOC dataset.
   
   b. Apply the finetuned BeRT and m-BeRT models to predict the labels (e.g., abuse or non-abuse) for the code-mixed sentences in the evaluation dataset.
   
   c. Calculate the accuracy of both models by comparing the predicted labels with the ground truth labels from the evaluation dataset.
   
   d. Compare the performance of BeRT and m-BeRT models at different levels of code mixing (varying CMI values) by analyzing the accuracy scores obtained.

5. Justification:

   a. Analyze and interpret the results to understand the relationship between code-mixing index (CMI) and model performance (accuracy).
   
   b. Justify the observations based on the hypothesis that as the CMI increases, the performance of the models might decrease due to the increased complexity and noise introduced by code mixing. Additionally, discuss the differences in performance between BeRT and m-BeRT models, considering the multilingual nature of m-BeRT.

Note: Thought of doing NLP along with Computer Vision, but couldn't get access to HASOC Dataset within time

# Section 2C - Computer Vision
To create a model that accepts a text prompt and an image as input and uses CLIP to create embeddings, followed by training a decoder for binary segmentation maps, these steps can be done:

1. **Data Preparation**: Downloaded the given dataset using `gdown` from the given link. Then resizing, normalizing, and converting them into suitable formats for training should be done. 

2. **Prepare the CLIP model**: Loading the pre-trained CLIP model from OpenAI's repository and initializing it. 

3. **Create embeddings**: Passing the text prompts and images through the CLIP model to obtain embeddings. Using the CLIP model's image and text encoders to create vector representations of the input.

4. **Train the decoder**: Designed a decoder architecture that takes the concatenated embeddings as input and produces a binary segmentation map. This decoder can be a convolutional neural network (CNN) or any suitable architecture that can map the embeddings to the segmentation map.
```
class Decoder(nn.Module):
    def __init__(self, embedding_size, segmentation_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(embedding_size, segmentation_size)

    def forward(self, embeddings):
        segmentation_map = self.fc(embeddings)
        return segmentation_map

embedding_size = 512  # Embedding size of CLIP model
segmentation_size = 256  # Size of the binary segmentation map

decoder = Decoder(embedding_size, segmentation_size)
decoder.to("cuda")
```
7. **Define loss function**: Can use pytorch's `nn.BCEWithLogitsLoss()` loss. This loss combines a Sigmoid layer and the BCELoss in one single class.
8. 
9. **Train the model**: Train the decoder using the embeddings obtained from the CLIP model. Pass the concatenated embeddings through the decoder and optimize the decoder parameters using backpropagation. Update the decoder's weights to minimize the defined loss function.

9. **Evaluate the model**: Use a validation set to evaluate the performance of the trained model. Calculate metrics such as accuracy, precision, recall, or intersection over union (IoU) to measure the quality of the binary segmentation maps produced by the model.

10. **Inference**: Use the trained model for inference by providing a text prompt and an image as input. Generate the embeddings using CLIP and pass them through the trained decoder to obtain the binary segmentation map for arbitrary objects specified in the text prompt.


Note: Extracted the zip file from gdown, but couldn't unzip it 
