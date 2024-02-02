# O output será a raiz quadrada da entrada, essa será a relação de entrada e saída na minha rede.
rm(list = ls())
input_train = as.data.frame(runif(1000, min = 0, max = 100))
output_train = sqrt(input_train)
# Criando A minha matriz de entradas e saídas e renomeando as colunas.
base_train = cbind(input_train, output_train)
colnames(base_train) = c("Input", "Output")
plot(base_train)


# Criando o modelo de neurônio e já treinando com meus dados de treino.
library(neuralnet)
classifier = neuralnet(Output ~ Input, base_train, hidden = c(10, 10), threshold = 0.1)
print(classifier)
plot(classifier)



# Criando os dados de teste para avaliar meu modelo.
base_test = as.data.frame((1:10)^2)
my_predict = compute(classifier, base_test)
print(my_predict$net.result)

tabel_1 = cbind(base_test, sqrt(base_test), 
              as.data.frame(my_predict$net.result))

