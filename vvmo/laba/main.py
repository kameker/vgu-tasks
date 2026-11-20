from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FactorAnalysis
from sklearn.metrics import mean_squared_error, pairwise_distances

folder = Path(__file__).resolve().parent
names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
df = pd.read_csv(folder / 'glass_raw.csv', names=names)
features = names[1:-1]
x = MinMaxScaler().fit_transform(df[features])
y = df['Type'].to_numpy()
colors = {1: 'blue', 2: 'orange', 3: 'green', 5: 'red', 6: 'purple', 7: 'brown'}

def scatter(ax, points):
    for label, color in colors.items():
        selected = y == label
        ax.scatter(points[selected, 0], points[selected, 1], s=18, alpha=0.7, color=color, label=str(label))

fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    scatter(ax, x[:, i:i + 2])
    ax.set_xlabel(features[i])
    ax.set_ylabel(features[i + 1])
fig.legend(*axes.flat[0].get_legend_handles_labels(), title='Класс', loc='outside upper center', ncols=6)
fig.savefig(folder / 'pairs.png', dpi=160)
plt.close(fig)

pca = PCA(n_components=2, svd_solver='full')
z = pca.fit_transform(x)
print('Объектов, признаков:', x.shape)
print('Классы:', df['Type'].value_counts().sort_index().to_dict())
print('PCA доли дисперсии, %:', np.round(pca.explained_variance_ratio_ * 100, 3))
print('PCA сумма долей, %:', round(pca.explained_variance_ratio_.sum() * 100, 3))
print('PCA собственные числа:', np.round(pca.explained_variance_, 6))
full = PCA().fit(x)
cumulative = np.cumsum(full.explained_variance_ratio_)
n = np.searchsorted(cumulative, 0.85) + 1
print('PCA накопленная дисперсия, %:', np.round(cumulative * 100, 2))
print('Компонент для 85%:', n)
print('PCA MSE восстановления (2 компоненты):', round(mean_squared_error(x, pca.inverse_transform(z)), 6))
for solver in ['full', 'randomized', 'arpack', 'auto']:
    model = PCA(n_components=2, svd_solver=solver, random_state=42)
    model.fit(x)
    print('PCA solver', solver, 'дисперсия %:', round(model.explained_variance_ratio_.sum() * 100, 4))

fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
scatter(axes[0, 0], z)
axes[0, 0].set_title('PCA')
for ax, kernel, gamma in zip(axes.flat[1:4], ['linear', 'rbf', 'rbf'], [None, 0.5, 5]):
    model = KernelPCA(n_components=2, kernel=kernel, gamma=gamma, fit_inverse_transform=True)
    reduced = model.fit_transform(x)
    restored = model.inverse_transform(reduced)
    print('KernelPCA', kernel, 'gamma', gamma, 'MSE:', round(mean_squared_error(x, restored), 6))
    if kernel == 'linear':
        difference = np.max(np.abs(pairwise_distances(z) - pairwise_distances(reduced)))
        print('PCA и linear KernelPCA разница расстояний:', round(difference, 10))
    scatter(ax, reduced)
    ax.set_title(f'KernelPCA {kernel}, gamma={gamma}')
for alpha in [0.1, 1]:
    model = SparsePCA(n_components=2, alpha=alpha, random_state=42)
    reduced = model.fit_transform(x)
    restored = reduced @ model.components_ + model.mean_
    print('SparsePCA alpha', alpha, 'MSE:', round(mean_squared_error(x, restored), 6), 'нулевых нагрузок:', np.sum(np.abs(model.components_) < 1e-8))
    if alpha == 0.1:
        scatter(axes[1, 1], reduced)
        axes[1, 1].set_title('SparsePCA alpha=0.1')
model = FactorAnalysis(n_components=2, random_state=42)
reduced = model.fit_transform(x)
restored = reduced @ model.components_ + model.mean_
print('FactorAnalysis MSE:', round(mean_squared_error(x, restored), 6))
scatter(axes[1, 2], reduced)
axes[1, 2].set_title('FactorAnalysis')
for ax in axes.flat:
    ax.set_xlabel('Компонента 1')
    ax.set_ylabel('Компонента 2')
fig.legend(*axes.flat[0].get_legend_handles_labels(), title='Класс', loc='outside upper center', ncols=6)
fig.savefig(folder / 'methods.png', dpi=160)
plt.close(fig)
