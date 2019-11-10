import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
wine = load_wine()
column_names = wine.feature_names

X = wine.data
y = wine.target



# print(f'sklearn datasets X shape: {X.shape}')
# print(f'sklearn datasets y shape: {y.shape}')
#
# print(f'keys: {wine.keys()}')
# print(f'data: {wine.data}')
# print(f'taget: {wine.target}')
# print(f'feature_names: {wine.feature_names}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
# print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

print(f'Intercept: {lm.intercept_}\n')

print(f'Coefficients: {lm.coef_}\n')
print(f'Named Coefficients: {pd.DataFrame(lm.coef_, column_names)}')

predicted_values = lm.predict(X_test)

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='inferno')

sns.scatterplot(y_test, predicted_values)
plt.plot([0, 50], [0, 50], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.show()

