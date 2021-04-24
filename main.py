import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def filterData(data):
	legendary = {False: 0, True: 1}

	data = data[['Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']]
	data = pd.get_dummies(data, columns=['Type 1', 'Type 2'], prefix=['type1', 'type2'])
	data['Legendary'] = [legendary[item] for item in data['Legendary']]

	return data

def normalizeData(data):
	for feature in data.columns:
		maxValue = data[feature].max()
		minValue = data[feature].min()
		data[feature] = (data[feature] - minValue) / (maxValue - minValue)

	return data

data = pd.read_csv('./data.csv')

data = filterData(data)
data = normalizeData(data)

print(data)