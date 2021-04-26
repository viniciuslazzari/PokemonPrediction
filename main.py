import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def filterData(data):
    legendary = {False: 0, True: 1}

    data = data[['Type 1', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']]
    data = pd.get_dummies(data, columns=['Type 1'], prefix=['type1'])
    data['Legendary'] = [legendary[item] for item in data['Legendary']]

    return data


def normalizeData(data):
    for feature in data.columns:
        maxValue = data[feature].max()
        minValue = data[feature].min()
        data[feature] = (data[feature] - minValue) / (maxValue - minValue)

    return data


def sigmoidFunction(x, theta):
    sigma = 1 / (1 + np.exp(- np.sum(x * theta, 1)))

    return sigma


def costFunction(y_pred, y):
    cost = - np.sum((y * np.log(y_pred)) + ((1 - y) * (np.log(1 - y_pred)))) / (len(y_pred))

    return cost


def gradientDescent(x, y, theta, alpha, iters):
    cost = []

    for iter in range(iters):
        y_pred = sigmoidFunction(x, theta)
        loss = y_pred - y
        for j in range(len(theta)):
            gradient = 0
            for m in range(len(x)):
                gradient += loss[m] * x[m][j]
            theta[j] -= (alpha/len(x)) * gradient

        print(costFunction(y_pred, y))
        cost.append(costFunction(y_pred, y))

    return theta, cost


def testModel(x, y, theta):
    y_pred = sigmoidFunction(x, theta)
    df = {'y_pred': y_pred, 'y': y}
    df = pd.DataFrame(df)
    df['y_pred'] = [round(item) for item in df['y_pred']]
    result = pd.concat([pd.DataFrame(x), df], axis=1)

    return df


data = pd.read_csv('./data.csv')

data = filterData(data)
data = normalizeData(data)

x = data.loc[:, data.columns != 'Legendary']
x.insert(0, 'coefficient', np.ones(len(data.index)))
y = data['Legendary']
theta = np.ones(len(x.columns))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
theta = np.array(theta).T

alpha = 0.003
iters = 10000

theta, cost = gradientDescent(x_train, y_train, theta, alpha, iters)

plt.plot(list(range(iters)), cost, '-r')
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()

test = testModel(x_test, y_test, theta)
score = accuracy_score(test['y'], test['y_pred'])
print(score)
