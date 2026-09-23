# Gerado por gerar_relatorio.py.
import numpy as np
import matplotlib.pyplot as plt

# Um único gerador compartilhado por todo o relatório.
rng = np.random.default_rng(42)
plt.rcParams.update({'figure.figsize': (8, 6), 'font.size': 11})

class_0 = rng.multivariate_normal([1.5, 1.5], [[0.5, 0], [0, 0.5]], size=1000)
class_1 = rng.multivariate_normal([5, 5], [[0.5, 0], [0, 0.5]], size=1000)
X = np.vstack((class_0, class_1))
y = np.concatenate((np.zeros(1000, dtype=int), np.ones(1000, dtype=int)))

# Auxiliares de apresentação reutilizados nas figuras dos dois exercícios.
def scatter_classes(ax, data, labels):
    for label, color in [(0, 'tab:blue'), (1, 'tab:orange')]:
        mask = labels == label
        ax.scatter(data[mask, 0], data[mask, 1], s=13, alpha=0.4,
                   color=color, label=f'Classe {label}')
    ax.set(xlabel='$x_1$', ylabel='$x_2$')
    ax.grid(alpha=0.2)

def boundary(ax, weights, bias, label, color, style='-'):
    # Desenha a reta sem ampliar os limites para valores fora da nuvem.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if not np.isclose(weights[1], 0):
        xx = np.array(xlim)
        ax.plot(xx, -(weights[0] * xx + bias) / weights[1],
                style, color=color, linewidth=2, label=label)
    elif not np.isclose(weights[0], 0):
        ax.axvline(-bias / weights[0], color=color, linestyle=style, label=label)
    ax.set(xlim=xlim, ylim=ylim)

fig, ax = plt.subplots()
scatter_classes(ax, X, y)
ax.set_title('Figura 1 — Dados separáveis (2000 amostras)')
ax.legend()
fig.tight_layout()

class Perceptron:
    def __init__(self, rng, learning_rate=0.01, n_iterations=100,
                 initial_weights=None, pocket=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        # O gerador é recebido, nunca reiniciado dentro da classe.
        self.weights = (rng.normal(0, 0.01, size=2) if initial_weights is None
                        else np.array(initial_weights, dtype=float).copy())
        self.initial_weights = self.weights.copy()
        self.bias = 0.0
        self.use_pocket = pocket

    def activation(self, z):
        return (np.asarray(z) >= 0).astype(int)

    def predict(self, x):
        return self.activation(np.dot(x, self.weights) + self.bias)

    def predict_all(self, X):
        return self.predict(X)

    def accuracy(self, X, y):
        return float(np.mean(self.predict_all(X) == y))

    def epoch(self, X, y):
        updates = 0
        for i in range(len(X)):
            y_pred = self.predict(X[i])
            error = y[i] - y_pred
            if error != 0:
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                updates += 1
                # Único acréscimo ao treino: avaliar e copiar após CADA atualização.
                if self.use_pocket:
                    accuracy = self.accuracy(X, y)
                    self.pocket_evaluations += 1
                    if accuracy > self.pocket_accuracy:
                        self.pocket_accuracy = accuracy
                        self.pocket_weights = self.weights.copy()
                        self.pocket_bias = float(self.bias)
                        self.pocket_epoch = self.current_epoch
                        self.pocket_sample = i + 1
        return updates

    def fit(self, X, y):
        self.accuracy_history = []
        self.updates_history = []
        self.pocket_history = []
        self.pocket_evaluations = 0
        if self.use_pocket:
            self.pocket_weights = self.weights.copy()
            self.pocket_bias = float(self.bias)
            self.pocket_accuracy = self.accuracy(X, y)
            self.pocket_epoch = 0
            self.pocket_sample = 0
        for epoch in range(1, self.n_iterations + 1):
            self.current_epoch = epoch
            updates = self.epoch(X, y)
            self.updates_history.append(updates)
            self.accuracy_history.append(self.accuracy(X, y))
            if self.use_pocket:
                self.pocket_history.append(self.pocket_accuracy)
            if updates == 0:
                break
        return self

    def metrics(self):
        return {'weights': self.weights.copy(), 'bias': self.bias,
                'num_epochs': len(self.accuracy_history),
                'final_accuracy': self.accuracy_history[-1],
                'accuracy_history': self.accuracy_history.copy()}

perceptron = Perceptron(rng, learning_rate=0.01, n_iterations=100)
perceptron.fit(X, y)
metrics = perceptron.metrics()
print('Pesos iniciais:', perceptron.initial_weights)
print('Pesos finais:', metrics['weights'])
print('Bias final:', metrics['bias'])
print('Épocas:', metrics['num_epochs'])
print('Acurácia final:', metrics['final_accuracy'])
print('Atualizações por época:', perceptron.updates_history)

fig, ax = plt.subplots()
scatter_classes(ax, X, y)
wrong = perceptron.predict_all(X) != y
ax.scatter(X[wrong, 0], X[wrong, 1], facecolors='none', edgecolors='black',
           s=65, label=f'Mal classificados ({wrong.sum()})')
boundary(ax, perceptron.weights, perceptron.bias, 'Fronteira final', 'black')
ax.set_title('Figura 2 — Fronteira nos dados separáveis')
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots()
epochs = np.arange(1, len(perceptron.accuracy_history) + 1)
ax.plot(epochs, perceptron.accuracy_history, marker='o', label='Acurácia (classes 0 e 1)')
ax.set(title='Figura 3 — Acurácia por época: dados separáveis',
       xlabel='Época', ylabel='Acurácia', ylim=(0, 1.05))
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()

# Mesmos dados, ordem, bias e pesos iniciais: muda apenas a taxa.
perceptron_eta_1 = Perceptron(rng, learning_rate=1.0, n_iterations=100,
                            initial_weights=perceptron.initial_weights)
perceptron_eta_1.fit(X, y)
direction_eta_001 = perceptron.weights / np.linalg.norm(perceptron.weights)
direction_eta_1 = perceptron_eta_1.weights / np.linalg.norm(perceptron_eta_1.weights)
print('eta=0.01:', perceptron.metrics())
print('eta=1.0:', perceptron_eta_1.metrics())
print('Direção eta=0.01:', direction_eta_001)
print('Direção eta=1.0:', direction_eta_1)

class_0_overlap = rng.multivariate_normal([3, 3], [[1.5, 0], [0, 1.5]], size=1000)
class_1_overlap = rng.multivariate_normal([4, 4], [[1.5, 0], [0, 1.5]], size=1000)
X2 = np.vstack((class_0_overlap, class_1_overlap))
y2 = np.concatenate((np.zeros(1000, dtype=int), np.ones(1000, dtype=int)))
fig, ax = plt.subplots()
scatter_classes(ax, X2, y2)
ax.set_title('Figura 4 — Dados sobrepostos (2000 amostras)')
ax.legend()
fig.tight_layout()

perceptron_overlap = Perceptron(rng, learning_rate=0.01, n_iterations=100, pocket=True)
perceptron_overlap.fit(X2, y2)
final_predictions = perceptron_overlap.predict_all(X2)
pocket_predictions = (X2 @ perceptron_overlap.pocket_weights +
                      perceptron_overlap.pocket_bias >= 0).astype(int)
print('Épocas:', len(perceptron_overlap.accuracy_history))
print('Final — w:', perceptron_overlap.weights, 'b:', perceptron_overlap.bias,
      'acurácia:', perceptron_overlap.accuracy(X2, y2))
print('Pocket — w:', perceptron_overlap.pocket_weights, 'b:', perceptron_overlap.pocket_bias,
      'acurácia:', perceptron_overlap.pocket_accuracy)
print('Melhor pocket — época:', perceptron_overlap.pocket_epoch,
      'amostra:', perceptron_overlap.pocket_sample)
print('Avaliações após atualizações:', perceptron_overlap.pocket_evaluations)
print('Última época — atualizações:', perceptron_overlap.updates_history[-1])

# Os dois painéis mostram as mesmas duas retas; separam somente as marcas de erro.
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
for ax, predictions, name in zip(axes, [final_predictions, pocket_predictions],
                                ['Final', 'Pocket']):
    scatter_classes(ax, X2, y2)
    wrong = predictions != y2
    ax.scatter(X2[wrong, 0], X2[wrong, 1], facecolors='none', edgecolors='black',
               linewidths=0.7, s=32, label=f'Erros {name.lower()} ({wrong.sum()})')
    boundary(ax, perceptron_overlap.weights, perceptron_overlap.bias,
             'Fronteira final', 'crimson', '--')
    boundary(ax, perceptron_overlap.pocket_weights, perceptron_overlap.pocket_bias,
             'Fronteira pocket', 'darkgreen')
    ax.set_title(f'{name}: {np.mean(predictions == y2):.2%} de acurácia')
    ax.legend(fontsize=9)
fig.suptitle('Figura 5 — Fronteiras final e pocket nos dados sobrepostos')
fig.tight_layout()

fig, ax = plt.subplots(figsize=(10, 5))
epochs2 = np.arange(1, len(perceptron_overlap.accuracy_history) + 1)
ax.plot(epochs2, perceptron_overlap.accuracy_history, color='crimson',
        label='Pesos atuais (classes 0 e 1)')
ax.plot(epochs2, perceptron_overlap.pocket_history, color='darkgreen',
        label='Melhor até agora — pocket (classes 0 e 1)')
ax.set(title='Figura 6 — Acurácia por época: dados sobrepostos',
       xlabel='Época', ylabel='Acurácia', ylim=(0.45, 0.8))
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()

# Diagnósticos para interpretar a fronteira e o efeito da ordem das amostras.
for name, prediction in [('Final', final_predictions), ('Pocket', pocket_predictions)]:
    matrix = np.array([[np.sum((y2 == real) & (prediction == predicted))
                        for predicted in (0, 1)] for real in (0, 1)])
    print(name, '— matriz (linhas reais, colunas preditas):', matrix.tolist())
mean_norm = float(np.linalg.norm(X2, axis=1).mean())
centers = np.array([[3, 3], [4, 4]])
print('Norma média de x:', mean_norm)
print('Escores finais nos centros:', centers @ perceptron_overlap.weights + perceptron_overlap.bias)
print('Escores pocket nos centros:', centers @ perceptron_overlap.pocket_weights + perceptron_overlap.pocket_bias)
print('Interseção final com x1=x2:', -perceptron_overlap.bias / perceptron_overlap.weights.sum())
print('Interseção pocket com x1=x2:', -perceptron_overlap.pocket_bias / perceptron_overlap.pocket_weights.sum())
plt.show()
