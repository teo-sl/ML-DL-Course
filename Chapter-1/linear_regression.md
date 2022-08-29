# Logistic regression and linear regression

## 1. Introduction to logistic r.

$$ y=mx+b$$

O, in un contesto multisimensionale

$$ y = w^t \cdot x +b $$

Che può essere espresso come

$$ y = \sum_i w_i w_i +b $$

Da ciò

$$ a = w^t \cdot x +b $$
if $a\ge 0 \rightarrow$ 1
if $a<0 \rightarrow 0$ 


Solitamente si usa una funzione di attivazione

$\hat{y} = \sigma(a)$ where, $a=w^t \cdot x$


Si può interpretare questa come una probabilità (nel caso che $\sigma$ sia una sigmoide)

Questo modello è detto logistic regression. Se:
- abbiamo una funzione lineare
- la sigmoide è applicata a tale funzione lineare

$$p(y=1 | x)=\sigma(w^T x +b)$$

        tf.keras.layers.Dense(output_size)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(D,)),
            tf.keras.layers.Dense(1,activation='sigmoid')
        ])

        loss='binary_crossentropy',
        metrics=['acccuracy']

        model.compile(optimizer='adam',
        loss='...'
        metrics=[])


        r=model.fit(X_train, y_train,
                    validation_data=(X_test,y_test),
                    epochs=100)


        plt.plot(r.history['loss'],label='loss')
        r.history['val_loss'],label='val_loss')
 
## 2. Second part (linear regression)

Potrebbe essere necessario effettuare delle trasformazioni sui dati.

 Nel caso della regressione, non serve funzione di attivazione, per tale motivo, non la si passa come argomento nella costruzione del modello.

 Nel caso del compile, si passa l'SGD, con  (0.001,0.9)

 Si può ridurre il lr in accordo al numero di epoche, verso la fine bisogna ridurlo per evitare che il modello non converga mai. lr schedule.

 ### 2.1 Loss

 Con la linear regression si usa il MSE

 $$ MSE = \frac{1}{N}\sum_i(y_i-\hat{y_i})^2$$ 

### 2.2 No accuracy 
Perché non ha senso verificare 0.0001 == 0.0002


### 2.3 Moore's law

$$C=A_0 r^t$$

t input, r rate of growth, A_0 valore iniziale.

Se si considerano i log della dimensione (y) questo si trasforma in una retta.

$$ log(C)=log(r)\cdot t+log(A_0)$$
