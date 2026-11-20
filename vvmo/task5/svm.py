import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from iris_data import variant3_x, variant3_y

x = np.array(variant3_x)
y = np.array(variant3_y)

model = SVC(kernel='linear')
model.fit(x, y)
predictions = model.predict(x)
errors = np.sum(predictions != y)

print(f'Число неверных классификаций: {errors}')
print(f'Процент неверных классификаций: {errors / len(y) * 100:.2f}%')

plt.scatter(x[y == -1, 0], x[y == -1, 1], color='blue', marker='o', label='Класс -1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='red', marker='^', label='Класс 1')

line_x = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
line_y = -(model.coef_[0, 0] * line_x + model.intercept_[0]) / model.coef_[0, 1]
plt.plot(line_x, line_y, color='black', label='Разделяющая линия')

plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.legend()
plt.grid()
plt.show()
