import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis = pd.read_csv('tennis_stats.csv')

print(tennis.head())
print(tennis.info())
print(tennis.describe())

# perform exploratory analysis here:
plt.scatter(tennis['Aces'], tennis['Wins'])
plt.show()
plt.close()

plt.scatter(tennis['DoubleFaults'], tennis['Losses'])
plt.show()
plt.close()

plt.scatter(tennis['BreakPointsOpportunities'], tennis['Winnings'])
plt.show()
plt.close()

## perform single feature linear regressions here:
x = tennis[['DoubleFaults']]
y = tennis[['Losses']]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8)

mdl = LinearRegression()

mdl.fit(x_train, y_train)

mdl.score(x_test,y_test)
y_predicted = mdl.predict(x_test)

plt.scatter(y_test, y_predicted)
plt.show()
plt.close()

#------------------------

x = tennis[['BreakPointsOpportunities']]
y = tennis[['Winnings']]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8)

mdl = LinearRegression()

mdl.fit(x_train, y_train)

mdl.score(x_test,y_test)
y_predicted = mdl.predict(x_test)

plt.scatter(y_test, y_predicted)
plt.show()
plt.close()

#-----------------------

x = tennis[['Aces']]
y = tennis[['Wins']]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8)

mdl = LinearRegression()

mdl.fit(x_train, y_train)

mdl.score(x_test,y_test)
y_predicted = mdl.predict(x_test)

plt.scatter(y_test, y_predicted)
plt.show()
plt.close()

## perform two feature linear regressions here:

x = tennis[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
y = tennis[['Winnings']]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8)

mdl = LinearRegression()

mdl.fit(x_train, y_train)

mdl.score(x_test,y_test)
y_predicted = mdl.predict(x_test)

plt.scatter(y_test, y_predicted, c='orange')
plt.show()
plt.close()

#--------------------------

x = tennis[['Aces', 'FirstServeReturnPointsWon']]
y = tennis[['Winnings']]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8)

mdl = LinearRegression()

mdl.fit(x_train, y_train)

mdl.score(x_test,y_test)
y_predicted = mdl.predict(x_test)

plt.scatter(y_test, y_predicted, c='green')
plt.show()
plt.close()

## perform multiple feature linear regressions here:

x = tennis[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
y = tennis[['Winnings']]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8)

mdl = LinearRegression()

mdl.fit(x_train, y_train)

mdl.score(x_test,y_test)
y_predicted = mdl.predict(x_test)

plt.scatter(y_test, y_predicted, c='violet')
plt.show()
plt.close()
