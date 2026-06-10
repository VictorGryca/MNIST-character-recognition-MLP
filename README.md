# Character recognition using Multi Layer Perceptrons and MNIST dataset

usar dropout? em que momento? 



--------------
estou tentando entender como funciona o backprop. ja entendi o forward pass, com as dimensoes dos tensores em cada camada e cada step das camadas e tmb onde cada funcao de ativacao vai. 

agora to tentando entender onde a funcao do SDG se encaixa, como faco para aplicar ela em cada peso, tambem estou tentando entender como calcular a funcao erro e se calculo a funcao do gradiente para cada peso e dai aplico a funcao para cada peso especifico dentro da camada (peso de cada neuronio). estou trabalhando na funcao do gradiente (w = w -Lr * dL/dw)

lembrei que o bias tmb precisa ser ajustado da mesma forma que os weights.

fui evoluindo no desenvolvimento das equacoes do backprop:
    fui quebrando o gradiente de w3 e w2. entendendo as derivadas parciais. comecando pelo output entendo o pq juntar cross-entropy com softmax. a derivada parcial fica bonita e facil. 
    

