import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

EXPECTED_COST = 0.00000003
ALPHA = 3
TEST_SIZE = 0.3

pd.set_option("display.max_rows", None, "display.max_columns", None)

# Função para filtrar os dados de treinamento
def filterData(data):
    # Dicionário para traduzir falso = 0 e verdadeiro = 1
    legendary_dict = {False: 0, True: 1}

    # Selecionando apenas as colunas desejadas
    data = data[['Type 1', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']]
    
    # Alterando o tipo de dado da coluna indicando a raridade de string para number
    data['Legendary'] = [legendary_dict[item] for item in data['Legendary']]

    # Realizando o processo de dummização do tipo primário do pokemon
    data = pd.get_dummies(data, columns=['Type 1'], prefix=['type1'])

    return data


def filterDataRef(ref, data):
    df = pd.DataFrame()

    data = pd.get_dummies(data, columns=['Type 1'], prefix=['type1'])
    df['Name'] = data['Name']

    for feature in ref.columns:
        df[feature] = data[feature] if feature in data.columns else np.zeros(len(data))

    return df

# Função para normalizar os dados de treinamento
def normalizeData(data):
    # Para cada atributo (coluna) do dataframe
    for feature in data.columns:
        # Obtem o maior e menor valor da coluna correspondente
        maxValue = data[feature].max()
        minValue = data[feature].min()

        # Realiza a normalização do dado atual, para que o mesmo sempre
        # corresponda a um valor entre 0 e 1
        data[feature] = (data[feature] - minValue) / (maxValue - minValue)

    return data


def normalizeDataRef(ref, data):
    for feature in ref.columns:
        if feature in data.columns and feature != 'Name':
            maxValue = ref[feature].max()
            minValue = ref[feature].min()
            data[feature] = (data[feature] - minValue) / (maxValue - minValue)

    return data

# Função para calcular sigmoid
def sigmoidFunction(x, theta):
    sigma = 1 / (1 + np.exp(- np.sum(x * theta, 1)))

    return sigma

# Função para calcular o custo
def costFunction(y_pred, y):
    cost = - np.sum((y * np.log(y_pred)) + ((1 - y) * (np.log(1 - y_pred)))) / (len(y_pred))

    return cost

# Função para realizar o gradiente
def gradientDescent(x, y, theta):
    # Vetor dos custos para plotar
    costArray = []
    # Booleano para controlar a convergência
    convergence = False

    while not convergence:
        # Calcula a saída com os parâmetros atuais
        y_pred = sigmoidFunction(x, theta)
        loss = y_pred - y
        for j in range(len(theta)):
            gradient = 0
            for m in range(len(x)):
                gradient += loss[m] * x[m][j]
            theta[j] -= (ALPHA/len(x)) * gradient

        cost = costFunction(y_pred, y)
        print(cost)
        costReduction = costArray[-1] - cost if costArray else cost
        costArray.append(cost)

        convergence = costReduction < EXPECTED_COST

    return theta, costArray


def testModel(x, y, theta):
    y_pred = sigmoidFunction(x, theta)
    df = {'y_pred': y_pred, 'y': y}
    df = pd.DataFrame(df)
    df['y_pred'] = [round(item) for item in df['y_pred']]

    return df


def predict(pred, theta):
    names = pred['Name']
    x = pred.loc[:, pred.columns != 'Name']
    x = np.array(x)
    y = sigmoidFunction(x, theta)
    y = [round(item, 2) for item in y]
    results = {'Name': names, 'Result': y}
    results = pd.DataFrame(results)

    return results

def main():
    data = pd.read_csv('./data.csv')
    pred = pd.read_csv('./predictions.csv')

    data = filterData(data)
    pred = normalizeDataRef(data, pred)
    data = normalizeData(data)
    pred = filterDataRef(data, pred)

    x_pred = pred.loc[:, pred.columns != 'Legendary']
    x_pred.insert(0, 'coefficient', np.ones(len(x_pred.index)))

    x = data.loc[:, data.columns != 'Legendary']
    x.insert(0, 'coefficient', np.ones(len(data.index)))
    y = data['Legendary']
    theta = np.ones(len(x.columns))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = TEST_SIZE, random_state = 0)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    theta = np.array(theta).T

    theta, cost = gradientDescent(x_train, y_train, theta)

    plt.plot(list(range(len(cost))), cost, '-r')
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()

    test = testModel(x_test, y_test, theta)
    score = accuracy_score(test['y'], test['y_pred'])
    print('The overall model cost is: ' + str(cost[-1]))
    print('The overall model accuracy is: ' + str(score))

    predictions = predict(x_pred, theta)
    print(predictions)

if __name__ == "__main__":
    main()