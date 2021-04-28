import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

legendary = {False: 0, True: 1}
pd.set_option("display.max_rows", None, "display.max_columns", None)


def filterData(data):
    data = data[['Type 1', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']]
    data['Legendary'] = [legendary[item] for item in data['Legendary']]
    data = pd.get_dummies(data, columns=['Type 1'], prefix=['type1'])

    return data


def filterDataRef(ref, data):
    df = pd.DataFrame()

    data = pd.get_dummies(data, columns=['Type 1'], prefix=['type1'])
    df['Name'] = data['Name']

    for feature in ref.columns:
        if feature in data.columns:
            df[feature] = data[feature]
        else:
            df[feature] = np.zeros(len(data))

    return df


def normalizeData(data):
    for feature in data.columns:
        maxValue = data[feature].max()
        minValue = data[feature].min()
        data[feature] = (data[feature] - minValue) / (maxValue - minValue)

    return data


def normalizeDataRef(ref, data):
    for feature in ref.columns:
        if feature in data.columns and feature != 'Name':
            maxValue = ref[feature].max()
            minValue = ref[feature].min()
            data[feature] = (data[feature] - minValue) / (maxValue - minValue)

    return data


def sigmoidFunction(x, theta):
    sigma = 1 / (1 + np.exp(- np.sum(x * theta, 1)))

    return sigma


def costFunction(y_pred, y):
    cost = - np.sum((y * np.log(y_pred)) + ((1 - y) * (np.log(1 - y_pred)))) / (len(y_pred))

    return cost


def gradientDescent(x, y, theta, alpha):
    costArray = []
    convergence = False

    while not convergence:
        y_pred = sigmoidFunction(x, theta)
        loss = y_pred - y
        for j in range(len(theta)):
            gradient = 0
            for m in range(len(x)):
                gradient += loss[m] * x[m][j]
            theta[j] -= (alpha/len(x)) * gradient

        cost = costFunction(y_pred, y)
        print(cost)
        costReduction = costArray[-1] - cost if costArray else cost
        costArray.append(cost)

        convergence = True if costReduction < 0.000000001 else False

    return theta, costArray


def testModel(x, y, theta):
    y_pred = sigmoidFunction(x, theta)
    df = {'y_pred': y_pred, 'y': y}
    df = pd.DataFrame(df)
    df['y_pred'] = [round(item) for item in df['y_pred']]
    result = pd.concat([pd.DataFrame(x), df], axis=1)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
theta = np.array(theta).T

alpha = 15

theta, cost = gradientDescent(x_train, y_train, theta, alpha)

plt.plot(list(range(len(cost))), cost, '-r')
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()

test = testModel(x_test, y_test, theta)
score = accuracy_score(test['y'], test['y_pred'])
print('The overall model accuracy is: ' + str(score))

predictions = predict(x_pred, theta)
print(predictions)
