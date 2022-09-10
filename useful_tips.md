

| task | activation function | # output neuron|Loss
|:-----|:--------------------|:--------------|---:|
|regression|None| -|mse|
|binary classification|Sigmoid|one neuron|binary_crossentropy|
|multiclass classification (#class=k)|softmax|k| 
|multiclass (tipo mnist)|- |- | sparse-cross-entropy|
|



Activation functions:

- sigmoid: binary classification
- none: regression
- softmax: multiclass classification


        