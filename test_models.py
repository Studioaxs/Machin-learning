import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import polynomial
from sklearn.datasets import make_regression
from my_models import LinearRegression, PolyRegression

# -------------------------------- Linear reg -------------------------------- #

x, y = make_regression(n_samples = 100, n_features = 1, noise = 10)
y = y.reshape(y.shape[0], 1) # Reshape y to (m, 1)
# print("y shape : ",y.shape)
model = PolyRegression()
hist = model.train(x,y, 5, 0.1, 1000)

plt.scatter(x,y)
plt.plot(x, model.predict(x), c='r')
plt.scatter(1,model.predict(1),c='g')
plt.legend(["Datas", "Model"])
plt.show()

# --------------------------- polynomial regression -------------------------- #
# Polynomial function
m = 100
n = 1
x = np.linspace(0, n, m).reshape(m, 1)
# print("x shape : ",x.shape)
y =  x**5 + x**4 + x**3 + x**2 + x + 0.1 * np.random.randn(m, 1)
# print("y shape : ",y.shape)
model = PolyRegression()
hist = model.train(x,y, 5, 0.1, 1000)

plt.scatter(x,y)
plt.plot(x, model.predict(x), c='r')
plt.scatter(1,model.predict(1),c='g')
plt.legend(["Datas", "Model"])
plt.show()

print(model.get_param())