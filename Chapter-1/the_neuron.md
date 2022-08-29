# The neuron and some tecnical stuff

* linear regression: regression
* logistic regression: classification



## linear regression

Semplicimente, trovare la retta che intercetta i punti.

$$ \hat{y} = m\cdot x +b$$

la retta che meglio intercetta i punti.

Il valore dei pesi, ci permette di dire quale sia l'importanza di un input nella predizione. 


## Come funziona l'apprendimento di un modello? 

Ci serve una funzione di errore:

$$ MSE = \frac{1}{N} \sum_i (y_i-\hat{y_i})^2 $$

Si può chiamare loss, error, cost.

L'obiettivo è trovare un punto dove la derivata è zero. Tuttavia, ci sono più punti del genere, e non è detto si tratti del minimo globale. 
Tensorflow fa tutto da lui.

### gradient descent
Non si può risolvere l'equazione in modo analitico. 

$$ w = w - \eta \nabla_w J$$
$$ b = b - \eta \nabla_b J$$

Si prendono piccoli passi nella direzione del gradiente. 

Come si sceglie $\eta$?
Soprattutto tramite trial and error.

## How to save a model

        model.save('blabla.h5')


        model=tf.keras.models.load_model('blabla.h5')

dovrebbe dare un errore con input layer esplicito, ovvero, mettendo tuttto nel .Sequential(....).
E non aggiungendo a parte i singoli pezzi.

E' stato sistemato







