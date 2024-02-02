rm(list = ls())
# Estou pegando 50 valores aleatórios do conjunto iris, todas as colunas estão inclusas.
train = iris[sample(1:nrow(iris), 50),]
base_train = train
print(train)


# Meu dataset base_train está sendo incrementado de 3 colunas que conterão se ela é de uma determinada espécie.
base_train = cbind(base_train, train$Species == 'setosa')
base_train = cbind(base_train, train$Species == 'versicolor')
base_train = cbind(base_train, train$Species == 'virginica')
print(base_train)


# Renomeando as colunas da minha rede neural.
names(base_train)[6] = 'setosa'
names(base_train)[7] = 'versicolor'
names(base_train)[8] = 'virginica'
head(base_train)

# Criando a rede neural através do pacote neuralnet.
library(neuralnet)
# Irei tentar predizer a planta tomando como base o seu formado e suas características.
classifier = neuralnet(setosa + versicolor + virginica ~
                         Sepal.Length + Sepal.Width + Petal.Length +
                         Petal.Width, data = base_train, hidden = c(6, 3))
                       
plot(classifier)
                       
                       
# Para mostrar o dataset iris.
iris_data_set = iris
# my_predict conterá o resultado do predict de todo o dataset iris.
# É válido falar que apenas 50 dados de entrada do dataset foram usados de treino.
my_predict = compute(classifier, iris[-5])$net.result
func = function(x) {
return(which(x == max(x)))
}
x = apply(my_predict, c(1), func)



# Aqui conterá os valores de cada predição do meu modelo de rede neural.
pred = c('setosa', 'versicolor', 'virginica')[x]
pred
library(caret)
library(ggplot2)
library(lattice)

# A matriz de confusão irá comparar as reais espécies com os valores previstos pelo meu modelo.
confusion_matrix = table(iris$Species, pred)
confusionMatrix(confusion_matrix)
                       
                       