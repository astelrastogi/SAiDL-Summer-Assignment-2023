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
