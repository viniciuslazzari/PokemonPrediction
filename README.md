# PokemonPrediction 🧠

Machine learning model to predict the rarity of pokemons using Python and multiple logistic regression.

## Data cleaning 📊

The original dataset has twelve independet variables: **Pokedex Number, Name, Type 1, Type 2, Total Stats, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Generation** and one output: **Legendary**.

|#  |Name                     |Type 1  |Type 2  |Total|HP |Attack|Defense|Sp. Atk|Sp. Def|Speed|Generation|Legendary|
|---|-------------------------|--------|--------|-----|---|------|-------|-------|-------|-----|----------|---------|
|1  |Bulbasaur                |Grass   |Poison  |318  |45 |49    |49     |65     |65     |45   |1         |False    |
|2  |Ivysaur                  |Grass   |Poison  |405  |60 |62    |63     |80     |80     |60   |1         |False    |
|3  |Venusaur                 |Grass   |Poison  |525  |80 |82    |83     |100    |100    |80   |1         |False    |
|3  |VenusaurMega Venusaur    |Grass   |Poison  |625  |80 |100   |123    |122    |120    |80   |1         |False    |

For this model, the variables **Pokedex Number, Name, Type 2, Total** and **Generation** will be ignored, since they are don't affect the final output.
**Type 1** is a dummy variable, hence the model does his treatment, by One-Hot Encoding them.

After this, since we will be using **gradient descent** to perform our training, the **data normalization** should be done, to avoid extra cost in the training algorithm.

|HP |Attack                   |Defense |Sp. Atk |Sp. Def|Speed|Legendary|type1_Bug|type1_Dark|type1_Dragon|type1_Electric|type1_Fairy|type1_Fighting|type1_Fire|type1_Flying|type1_Ghost|type1_Grass|type1_Ground|type1_Ice|type1_Normal|type1_Poison|type1_Psychic|type1_Rock|type1_Steel|type1_Water|
|---|-------------------------|--------|--------|-------|-----|---------|---------|----------|------------|--------------|-----------|--------------|----------|------------|-----------|-----------|------------|---------|------------|------------|-------------|----------|-----------|-----------|
|0.1732283464566929|0.23783783783783785      |0.19555555555555557|0.29891304347826086|0.21428571428571427|0.22857142857142856|0.0      |0.0      |0.0       |0.0         |0.0           |0.0        |0.0           |0.0       |0.0         |0.0        |1.0        |0.0         |0.0      |0.0         |0.0         |0.0          |0.0       |0.0        |0.0        |
|0.23228346456692914|0.3081081081081081       |0.2577777777777778|0.3804347826086957|0.2857142857142857|0.3142857142857143|0.0      |0.0      |0.0       |0.0         |0.0           |0.0        |0.0           |0.0       |0.0         |0.0        |1.0        |0.0         |0.0      |0.0         |0.0         |0.0          |0.0       |0.0        |0.0        |
|0.3110236220472441|0.41621621621621624      |0.3466666666666667|0.4891304347826087|0.38095238095238093|0.42857142857142855|0.0      |0.0      |0.0       |0.0         |0.0           |0.0        |0.0           |0.0       |0.0         |0.0        |1.0        |0.0         |0.0      |0.0         |0.0         |0.0          |0.0       |0.0        |0.0        |
|0.3110236220472441|0.5135135135135135       |0.5244444444444445|0.6086956521739131|0.47619047619047616|0.42857142857142855|0.0      |0.0      |0.0       |0.0         |0.0           |0.0        |0.0           |0.0       |0.0         |0.0        |1.0        |0.0         |0.0      |0.0         |0.0         |0.0          |0.0       |0.0        |0.0        |

And know the data is ready to be used, it just needs to be separated between the training and testing dataset, what can be done using **sklearn** function `train_test_split`.

## Model training 🔄

In logistic regression the output should not be a continuous value, but one that ranges between 0 (False) and 1 (True). Hence, the output is calculated using a the **sigmoid function**, that ensures that the final output will be some value between 0 and 1, no matter how big the theta parameters are.

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e376fe69caee24c914fab1360de36900b7bb9c24">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sigmoid-function-2.svg/1280px-Sigmoid-function-2.svg.png">

The cost function for logistic regression is very different than those of linear regression, in fact, there are two cost functions, one when `y = 0` and one when `y = 1`. They could be joined together to get the general cost function:

<img src="https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-91ede8271c1696055f1ed51e65ff05f2_l3.svg">

This way, the cost increases exponentially, because he derives from the log function, that punishes bigger errors with bigger costs.

<img src="https://miro.medium.com/max/709/1*gAsyT-YdsQZUMF81NTZQdQ.png">

<img src="https://miro.medium.com/max/695/1*2QLAi8r4BWFZ4AC6aQLzbA.png">

For every iteration, the cost function is calculated and should be minimized ultil convergence, a state where almost nothing changes anymore. To minimize the function, the model uses the derivative of **gradient descent** for each `theta` parameter, and update all of them simultaneously.

<img src="https://render.githubusercontent.com/render/math?math=\displaystyle \theta_j := \theta_j - \frac{1}{m} \alpha \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}_j">


This way, if everything is working and a good learning rate `α` is chosen, the **cost** should decrease in every iteration, improving the model accuracy.

<img src="https://i.postimg.cc/N0QJfcYn/cost.png">

## Model testing 📚

The test of the model was made using the dataset that was initially separated from the training set. For the test, the hypothesis of each sample was calculated, based on the `theta` values generated from training. Then, the algorithm calculates the overallmodel accuracy using **sklearn** function `accuracy_score`.

### Results

Training the algorithm with `α = 0.003` and repeating the process until convergence, the overall model accuracy was **92%**.

## How to make predictions 🔮

Once the model is trained, any pokemon can be created and send to the algorithm, that will predict the probability of him being **Legendary**.
The only thing that you have to do is to edit the `predictions` file, creating a new line and adding the pokemon stats. The next time that the algorithm run, the pokemon created will be printed in terminal, along with his probability.

## Technologies 💻

Build with:
- [Python](https://www.python.org/)
- [pandas](https://github.com/pandas-dev/pandas)
- [NumPy](https://github.com/numpy/numpy)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)

## Author 🧙
- [viniciuslazzari](https://github.com/viniciuslazzari)