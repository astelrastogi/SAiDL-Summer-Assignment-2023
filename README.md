# SAiDL-Summer-Assignment-2023
Repository for submitting solutions to SAiDL Summer Assignment 2023. Implemented Section 1 along with its bonus task, and Section 2 and 3. 


# Section 1

The code is present [here](). First use a standard Softmax, and then a Gumbel-Softmax function. 

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

