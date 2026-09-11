#!/usr/bin/env python3
# Executar: python docs/exercises/data/code/analise.py
# Fonte científica incluída diretamente no relatório por pymdownx.snippets.
# Para recalcular também a narrativa: execute code/gerar_relatorio.py.

# --8<-- [start:bloco_01]
from pathlib import Path
from itertools import combinations
import hashlib
import json
import platform
import os

# Caminhos independentes da pasta atual: funciona em um clone limpo.
if "__file__" in globals():
    EXERCISE_DIR = Path(__file__).resolve().parent.parent
else:
    # Jupyter: aberto em code/, na pasta do exercício ou na raiz do repositório.
    candidatos = [Path.cwd().parent, Path.cwd(), Path.cwd() / "docs/exercises/data"]
    EXERCISE_DIR = next(p for p in candidatos if (p / "data/train.csv").is_file())
os.environ.setdefault("MPLCONFIGDIR", str(EXERCISE_DIR.parents[2] / ".cache/matplotlib"))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Único gerador: não reinicializar nem criar outras sementes nos exercícios.
rng = np.random.default_rng(42)
OUT = EXERCISE_DIR / "resultados"
FIG = EXERCISE_DIR / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
CORES = ["#2463A6", "#D97720", "#269477", "#9656A6"]
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 180, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.17, "legend.framealpha": 0.95,
})

def salvar_figura(fig, numero):
    fig.savefig(FIG / f"figura_{numero}.png", bbox_inches="tight", facecolor="white")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

versoes = {"Python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
           "scikit-learn": sklearn.__version__}
print("Versões usadas:", versoes)
# --8<-- [end:bloco_01]

# --8<-- [start:bloco_02]
medias = np.array([[2, 3], [5, 6], [8, 1], [15, 4]], dtype=float)
desvios = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
y_2d = np.repeat(np.arange(4), 100)

def gerar_nuvens(s):
    return np.vstack([rng.normal(mu, s * sd, size=(100, 2))
                      for mu, sd in zip(medias, desvios)])

X_original = gerar_nuvens(1.0)
assert X_original.shape == (400, 2)
assert np.array_equal(np.bincount(y_2d), [100, 100, 100, 100])

def desenhar_nuvens(ax, X, centros=True):
    for k, cor in enumerate(CORES):
        ax.scatter(*X[y_2d == k].T, s=22, color=cor, alpha=0.72,
                   label=f"Classe {k}", linewidths=0)
        if centros:
            ax.scatter(*medias[k], s=160, marker="X", c=cor,
                       edgecolor="black", linewidth=1.1, zorder=5)
    ax.set_xlabel("Feature x")
    ax.set_ylabel("Feature y")
    ax.set_aspect("equal", adjustable="box")

fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
desenhar_nuvens(ax, X_original)
xlim = (X_original[:, 0].min() - 1, X_original[:, 0].max() + 1)
ylim = (X_original[:, 1].min() - 1, X_original[:, 1].max() + 1)
gx, gy = np.meshgrid(np.linspace(*xlim, 550), np.linspace(*ylim, 450))
grade = np.column_stack([gx.ravel(), gy.ravel()])
# Esboço pelas densidades gaussianas conhecidas e priors iguais.
# Menor escore = maior densidade. Não há ajuste ou treinamento.
escores = (((grade[:, None, :] - medias) / desvios) ** 2).sum(axis=2)
escores += 2 * np.log(desvios).sum(axis=1)
regioes = escores.argmin(axis=1).reshape(gx.shape)
for k in range(4):
    ax.contour(gx, gy, (regioes == k).astype(float), levels=[0.5],
               colors=["#333333"], linewidths=1, linestyles="--", alpha=0.75)
ax.set(xlim=xlim, ylim=ylim, title="Figura 1 — Nuvens originais e esboço de fronteiras (s = 1)")
handles, labels = ax.get_legend_handles_labels()
handles += [Line2D([], [], marker="X", color="black", linestyle="None", markersize=9),
            Line2D([], [], color="#333333", linestyle="--")]
ax.legend(handles, labels + ["Média especificada", "Esboço: densidades conhecidas"],
          loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
salvar_figura(fig, 1)
# --8<-- [end:bloco_02]

# --8<-- [start:bloco_03]
escalas = np.array([0.5, 1.0, 2.0, 4.0])
datasets_2d = {1.0: X_original}
for s in (0.5, 2.0, 4.0):
    datasets_2d[s] = gerar_nuvens(s)

pares = list(combinations(range(4), 2))
sd_medio = desvios.mean(axis=1)
rij = pd.DataFrame([
    {"Par": f"({i}, {j})", "Distância dos centros": np.linalg.norm(medias[i] - medias[j]),
     "Soma dos desvios médios": sd_medio[i] + sd_medio[j],
     "r_ij em s=1": np.linalg.norm(medias[i] - medias[j]) / (sd_medio[i] + sd_medio[j])}
    for i, j in pares
])
menor_linha = rij.loc[rij["r_ij em s=1"].idxmin()]
rmin = float(menor_linha["r_ij em s=1"])
par_min = menor_linha["Par"]

taxas, erros_mistura, taxas_classe = {}, {}, {}
for s in escalas:
    X = datasets_2d[s]
    d2 = ((X[:, None, :] - medias[None, :, :]) ** 2).sum(axis=2)
    diferente = d2.argmin(axis=1) != y_2d
    taxas[s] = diferente.mean()
    erros_mistura[s] = int(diferente.sum())
    taxas_classe[s] = [diferente[y_2d == k].mean() for k in range(4)]
mistura = pd.DataFrame({"s": escalas,
                       "Pontos misturados / 400": [erros_mistura[s] for s in escalas],
                       "Taxa de mistura (%)": [100 * taxas[s] for s in escalas]})

# Mesmos limites, mesma proporção x:y e todas as observações visíveis.
todos = np.vstack([datasets_2d[s] for s in escalas])
limites = np.column_stack([todos.min(axis=0) - 1, todos.max(axis=0) + 1])
fig, axes = plt.subplots(2, 2, figsize=(8.7, 10.6), sharex=True, sharey=True,
                         layout="compressed")
for ax, s in zip(axes.ravel(), escalas):
    desenhar_nuvens(ax, datasets_2d[s])
    ax.set(xlim=limites[0], ylim=limites[1],
           title=f"s = {s:.1f} | mistura = {100 * taxas[s]:.2f}%")
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle("Figura 2 — Espalhamento com eixos compartilhados", fontsize=13)
salvar_figura(fig, 2)

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for k, cor in enumerate(CORES):
    ax.plot(escalas, [100 * taxas_classe[s][k] for s in escalas],
            "o--", color=cor, alpha=0.8, label=f"Classe {k}")
total = np.array([100 * taxas[s] for s in escalas])
ax.plot(escalas, total, "o-", color="#222222", lw=2.3,
        label="Total: 4 classes (mesmo peso)")
for s, valor in zip(escalas, total):
    ax.annotate(f"{valor:.2f}%", (s, valor), xytext=(3, 10),
                textcoords="offset points", fontsize=9, fontweight="bold")
ax.set(title="Figura 3 — Taxa de mistura × fator de escala",
       xlabel="Fator de escala s", ylabel="Pontos com centro mais próximo de outra classe (%)",
       xticks=escalas, ylim=(-2, max(70, total.max() + 15)))
ax.legend(fontsize=9)
salvar_figura(fig, 3)
print(rij.to_string(index=False))
print(mistura.to_string(index=False))
# --8<-- [end:bloco_03]

# --8<-- [start:bloco_04]
def fecho_convexo(pontos):
    # Vértices ordenados do menor polígono convexo que contém os pontos 2D.
    pts = sorted(set(map(tuple, pontos)))
    def cruz(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    inferior, superior = [], []
    for p in pts:
        while len(inferior) >= 2 and cruz(inferior[-2], inferior[-1], p) <= 0:
            inferior.pop()
        inferior.append(p)
    for p in reversed(pts):
        while len(superior) >= 2 and cruz(superior[-2], superior[-1], p) <= 0:
            superior.pop()
        superior.append(p)
    return np.asarray(inferior[:-1] + superior[:-1])

def fechos_intersectam(a, b, tol=1e-10):
    # Teorema dos eixos separadores para dois polígonos convexos.
    for poligono in (a, b):
        arestas = np.roll(poligono, -1, axis=0) - poligono
        normais = np.column_stack([-arestas[:, 1], arestas[:, 0]])
        normais /= np.linalg.norm(normais, axis=1, keepdims=True)
        pa, pb = a @ normais.T, b @ normais.T
        separado = ((pa.max(axis=0) < pb.min(axis=0) - tol) |
                    (pb.max(axis=0) < pa.min(axis=0) - tol))
        if separado.any():
            return False
    return True

intersecoes = {}
for s in escalas:
    fechos = [fecho_convexo(datasets_2d[s][y_2d == k]) for k in range(4)]
    intersecoes[s] = [(i, j) for i, j in pares
                      if fechos_intersectam(fechos[i], fechos[j])]
primeiro_s = next(s for s in escalas if intersecoes[s])
separabilidade = pd.DataFrame([
    {"s": s, "Pares com fechos sobrepostos": str(intersecoes[s]) if intersecoes[s] else "Nenhum",
     "Todos os pares separáveis por retas?": "Não" if intersecoes[s] else "Sim"}
    for s in escalas
])
print(separabilidade.to_string(index=False))
# --8<-- [end:bloco_04]

# --8<-- [start:bloco_05]
mu_A = np.zeros(5)
mu_B = np.full(5, 1.5)
cov_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
cov_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])
assert np.linalg.eigvalsh(cov_A).min() > 0
assert np.linalg.eigvalsh(cov_B).min() > 0
X_A = rng.multivariate_normal(mu_A, cov_A, size=500, method="cholesky")
X_B = rng.multivariate_normal(mu_B, cov_B, size=500, method="cholesky")
X_I = np.vstack([X_A, X_B])
y_I = np.repeat([0, 1], 500)
print("Dataset I:", X_I.shape, "; contagens:", np.bincount(y_I))
# --8<-- [end:bloco_05]

# --8<-- [start:bloco_06]
def gerar_casca(media_raio):
    v = rng.normal(size=(500, 5))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    rho = rng.normal(loc=media_raio, scale=0.4, size=500)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    # Sem truncamento: mantém exatamente a distribuição pedida.
    return rho[:, None] * u, rho

X_C, rho_C = gerar_casca(2.0)
X_D, rho_D = gerar_casca(5.0)
X_II = np.vstack([X_C, X_D])
y_II = np.repeat([0, 1], 500)
print("Dataset II:", X_II.shape, "; contagens:", np.bincount(y_II))
print("Raios sorteados negativos:", int((rho_C < 0).sum() + (rho_D < 0).sum()))
# --8<-- [end:bloco_06]

# --8<-- [start:bloco_07]
pca_I = PCA(n_components=2, svd_solver="full")
pca_II = PCA(n_components=2, svd_solver="full")
Z_I = pca_I.fit_transform(X_I)
Z_II = pca_II.fit_transform(X_II)
dist_I = float(np.linalg.norm(X_A.mean(axis=0) - X_B.mean(axis=0)))
dist_II = float(np.linalg.norm(X_C.mean(axis=0) - X_D.mean(axis=0)))
dist_I_teorica = float(np.linalg.norm(mu_A - mu_B))
variancia_I = pca_I.explained_variance_ratio_
variancia_II = pca_II.explained_variance_ratio_
raios_I = np.linalg.norm(X_I, axis=1)
raios_II = np.linalg.norm(X_II, axis=1)
metricas_5d = pd.DataFrame({
    "Dataset": ["I", "II"],
    "Distância amostral (5D)": [dist_I, dist_II],
    "Distância populacional": [dist_I_teorica, 0.0],
    "PC1 (%)": [100 * variancia_I[0], 100 * variancia_II[0]],
    "PC2 (%)": [100 * variancia_I[1], 100 * variancia_II[1]],
    "PC1 + PC2 (%)": [100 * variancia_I.sum(), 100 * variancia_II.sum()],
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharex=True, sharey=True,
                         layout="constrained")
lim_pca = max(np.abs(Z_I).max(), np.abs(Z_II).max()) + 0.4
for ax, Z, y, nomes, titulo, var in [
    (axes[0], Z_I, y_I, ["A", "B"], "I — Gaussianas deslocadas", variancia_I),
    (axes[1], Z_II, y_II, ["C (núcleo)", "D (casca)"], "II — Cascas concêntricas", variancia_II),
]:
    for k, nome in enumerate(nomes):
        ax.scatter(*Z[y == k].T, c=CORES[k], s=14, alpha=0.5,
                   linewidths=0, label=f"Classe {nome}")
    ax.set(title=f"{titulo}\nPC1 + PC2: {100*var.sum():.2f}%",
           xlabel=f"PC1 ({100*var[0]:.2f}%)", ylabel=f"PC2 ({100*var[1]:.2f}%)",
           xlim=(-lim_pca, lim_pca), ylim=(-lim_pca, lim_pca))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelleft=True)
    ax.legend(loc="upper left", fontsize=9)
fig.suptitle("Figura 4 — Projeções PCA de 5D para 2D", fontsize=15)
salvar_figura(fig, 4)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True,
                         layout="constrained")
bins = np.linspace(0, max(raios_I.max(), raios_II.max()) + 0.1, 38)
for ax, raios, y, nomes, titulo in [
    (axes[0], raios_I, y_I, ["A", "B"], "I — Gaussianas deslocadas"),
    (axes[1], raios_II, y_II, ["C (núcleo)", "D (casca)"], "II — Cascas concêntricas"),
]:
    for k, nome in enumerate(nomes):
        ax.hist(raios[y == k], bins=bins, color=CORES[k], alpha=0.58,
                label=f"Classe {nome}", edgecolor="white", linewidth=0.4)
    ax.set(title=titulo, xlabel=r"Raio em 5D: $\|x\|_2$ (distância à origem)",
           ylabel="Número de pontos")
    ax.tick_params(labelleft=True)
    ax.legend()
axes[1].axvline(3.5, color="#333333", ls="--", label="Limiar proposto: 3,5")
axes[1].legend()
fig.suptitle("Figura 5 — Distribuição dos raios no espaço original (5D)", fontsize=15)
salvar_figura(fig, 5)
print(metricas_5d.to_string(index=False))
# --8<-- [end:bloco_07]

# --8<-- [start:bloco_08]
def regra_radial(X):
    # Regra analítica, sem fit: 0 = C; 1 = D.
    return ((X ** 2).sum(axis=1) >= 3.5 ** 2).astype(int)

erros_radiais = int((regra_radial(X_II) != y_II).sum())
intervalos_raio = pd.DataFrame([
    {"Classe": nome, "Raio mínimo": np.linalg.norm(X, axis=1).min(),
     "Raio médio": np.linalg.norm(X, axis=1).mean(),
     "Raio máximo": np.linalg.norm(X, axis=1).max()}
    for nome, X in [("A", X_A), ("B", X_B), ("C", X_C), ("D", X_D)]
])
print(intervalos_raio.to_string(index=False))
print("Erros da regra radial na amostra do Dataset II:", erros_radiais, "/ 1000")

# Exemplo geométrico adicional: 11 fica entre o maior raio ao quadrado de C
# e o menor de D nesta amostra. Não é um desempenho de generalização.
def separador_da_amostra(X):
    return ((X ** 2).sum(axis=1) >= 11.0).astype(int)

maior_raio_C = float(np.linalg.norm(X_C, axis=1).max())
menor_raio_D = float(np.linalg.norm(X_D, axis=1).min())
assert maior_raio_C**2 < 11.0 < menor_raio_D**2
erros_separador_amostra = int((separador_da_amostra(X_II) != y_II).sum())
print("Erros com o limiar ilustrativo ||x||² = 11:", erros_separador_amostra, "/ 1000")
# --8<-- [end:bloco_08]

# --8<-- [start:bloco_09]
arquivo = EXERCISE_DIR / "data/train.csv"
hash_esperado = "17336d553f49ebdf6ecb266d2b5d3746e5dd308445f7c7864141c4f28d2a88d0"
hash_obtido = hashlib.sha256(arquivo.read_bytes()).hexdigest()
assert hash_obtido == hash_esperado, "Arquivo diferente da cópia usada no relatório."
df = pd.read_csv(arquivo)
assert df.shape == (8693, 14)
assert df["PassengerId"].is_unique
assert not df["Transported"].isna().any()
y_real = df["Transported"].astype(int)

def split_estratificado(y, proporcao_teste=0.2):
    # Estratificação com o MESMO rng; sklearn fica restrito à PCA e ao
    # pré-processamento. Maiores restos resolvem o arredondamento das cotas.
    classes, contagens = np.unique(y, return_counts=True)
    cotas = proporcao_teste * contagens
    n_testes = np.floor(cotas).astype(int)
    total_teste = int(np.ceil(len(y) * proporcao_teste))
    faltam = total_teste - n_testes.sum()
    ordem_restos = np.argsort(-(cotas - n_testes), kind="stable")
    n_testes[ordem_restos[:faltam]] += 1
    treino, teste = [], []
    for classe, n_teste in zip(classes, n_testes):
        indices = rng.permutation(np.flatnonzero(np.asarray(y) == classe))
        teste.extend(indices[:n_teste])
        treino.extend(indices[n_teste:])
    return rng.permutation(treino), rng.permutation(teste)

# A reserva ocorre antes de qualquer imputação, codificação ou escala.
idx_treino, idx_teste = split_estratificado(y_real)
treino_bruto = df.iloc[idx_treino].copy()
teste_bruto = df.iloc[idx_teste].copy()
y_treino = y_real.iloc[idx_treino].to_numpy()
y_teste = y_real.iloc[idx_teste].to_numpy()

gastos = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
numericas_originais = ["Age"] + gastos
categoricas = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
balanceamento = pd.DataFrame({
    "Transported": ["False (0)", "True (1)"],
    "Contagem": [(y_real == 0).sum(), (y_real == 1).sum()],
    "Percentual (%)": [100 * (y_real == 0).mean(), 100 * (y_real == 1).mean()],
})
faltantes = pd.DataFrame({"Coluna": df.columns,
                         "Faltantes": df.isna().sum().to_numpy(),
                         "Percentual (%)": 100 * df.isna().mean().to_numpy()})
estat_gastos = df[gastos].agg(["mean", "median", "max"]).T.reset_index()
estat_gastos.columns = ["Coluna", "Média", "Mediana", "Máximo"]
print("SHA-256:", hash_obtido)
print(balanceamento.to_string(index=False))
print(faltantes.to_string(index=False))
print(estat_gastos.to_string(index=False))
# --8<-- [end:bloco_09]

# --8<-- [start:bloco_10]
assert len(np.intersect1d(idx_treino, idx_teste)) == 0
assert np.array_equal(np.sort(np.r_[idx_treino, idx_teste]), np.arange(len(df)))
split_tabela = pd.DataFrame([
    {"Conjunto": nome, "Amostras": len(y), "False (0)": int((y == 0).sum()),
     "True (1)": int((y == 1).sum()), "Positivos (%)": 100 * y.mean()}
    for nome, y in [("Treino", y_treino), ("Teste", y_teste)]
])
food_media_treino = float(treino_bruto["FoodCourt"].mean())
food_mediana_treino = float(treino_bruto["FoodCourt"].median())
print(split_tabela.to_string(index=False))
print("FoodCourt no treino bruto — média:", food_media_treino,
      "; mediana:", food_mediana_treino)
# --8<-- [end:bloco_10]

# --8<-- [start:bloco_11]
# 1. Imputação numérica: estatísticas aprendidas somente no treino.
imputador_num = SimpleImputer(strategy="median")
treino_num = pd.DataFrame(
    imputador_num.fit_transform(treino_bruto[numericas_originais]),
    columns=numericas_originais, index=treino_bruto.index,
)
teste_num = pd.DataFrame(
    imputador_num.transform(teste_bruto[numericas_originais]),
    columns=numericas_originais, index=teste_bruto.index,
)
medianas_imputacao = pd.DataFrame({"Coluna": numericas_originais,
                                  "Mediana aprendida no treino": imputador_num.statistics_})

# 2. Soma em unidades monetárias originais, depois da imputação.
for frame in (treino_num, teste_num):
    frame["TotalSpend"] = frame[gastos].sum(axis=1)
numericas = numericas_originais + ["TotalSpend"]
gastos_com_total = gastos + ["TotalSpend"]
assert (treino_num[gastos_com_total] >= 0).all().all()
assert (teste_num[gastos_com_total] >= 0).all().all()

# 3. Log nos gastos individuais e totais; Age continua sem log.
treino_log, teste_log = treino_num.copy(), teste_num.copy()
treino_log[gastos_com_total] = np.log1p(treino_num[gastos_com_total])
teste_log[gastos_com_total] = np.log1p(teste_num[gastos_com_total])

# 4. Min-max: um único fit, só no treino. Teste usa apenas transform.
escalonador = MinMaxScaler(feature_range=(-1, 1), clip=True)
treino_escalado = escalonador.fit_transform(treino_log)
teste_escalado = escalonador.transform(teste_log)
# Diagnóstico de clipping com os mesmos parâmetros aprendidos.
teste_sem_clip = teste_log.to_numpy() * escalonador.scale_ + escalonador.min_
clip_mask = (teste_sem_clip < -1 - 1e-12) | (teste_sem_clip > 1 + 1e-12)
clip_colunas = pd.DataFrame({"Coluna": numericas,
                             "Células do teste truncadas": clip_mask.sum(axis=0)})

# 5. Tipo uniforme por coluna categórica, preservando faltantes como np.nan.
def preparar_categoricas(frame):
    cat = frame[categoricas].astype("string").astype(object)
    return cat.where(pd.notna(cat), np.nan)

imputador_cat = SimpleImputer(strategy="constant", fill_value="Ausente")
treino_cat = imputador_cat.fit_transform(preparar_categoricas(treino_bruto))
teste_cat = imputador_cat.transform(preparar_categoricas(teste_bruto))
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
treino_onehot = encoder.fit_transform(treino_cat)
teste_onehot = encoder.transform(teste_cat)
nomes_onehot = encoder.get_feature_names_out(categoricas).tolist()

# 6. Matrizes finais: sete colunas numéricas e os blocos one-hot.
nomes_features = numericas + nomes_onehot
X_treino_final = np.column_stack([treino_escalado, treino_onehot])
X_teste_final = np.column_stack([teste_escalado, teste_onehot])
categorias_aprendidas = pd.DataFrame({
    "Feature": categoricas,
    "Categorias vistas no treino": [", ".join(c) for c in encoder.categories_],
    "Número de colunas one-hot": [len(c) for c in encoder.categories_],
})
ineditas_teste = {
    col: sorted(set(teste_cat[:, j]) - set(encoder.categories_[j]))
    for j, col in enumerate(categoricas)
}

# Verifica o contrato para uma categoria inédita sem mudar nem reajustar dados.
exemplo_inedito = teste_cat[:1].copy()
exemplo_inedito[0, 0] = "Planeta_inedito"
codificado_inedito = encoder.transform(exemplo_inedito)
assert (codificado_inedito[0, :len(encoder.categories_[0])] == 0).all()

intervalos_finais = pd.DataFrame([
    {"Conjunto": nome, "Mínimo numérico": Xn.min(), "Máximo numérico": Xn.max(),
     "Mínimo da matriz final": X.min(), "Máximo da matriz final": X.max()}
    for nome, Xn, X in [("Treino", treino_escalado, X_treino_final),
                        ("Teste", teste_escalado, X_teste_final)]
])
print(medianas_imputacao.to_string(index=False))
print(categorias_aprendidas.to_string(index=False))
print("Categorias inéditas no teste real:", ineditas_teste)
print(clip_colunas.to_string(index=False))
print(intervalos_finais.to_string(index=False))
# --8<-- [end:bloco_11]

# --8<-- [start:bloco_12]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), layout="constrained")
food_j = numericas.index("FoodCourt")
painel_dados = [
    (treino_bruto["FoodCourt"].to_numpy(), "Antes: valores observados", "FoodCourt (gasto original)"),
    (treino_log["FoodCourt"].to_numpy(), "Após imputação e log1p", "log(1 + FoodCourt)"),
    (treino_escalado[:, food_j], "Depois: entrada final da rede", "FoodCourt normalizado em [−1, 1]"),
]
for ax, (valores, titulo, rotulo) in zip(axes, painel_dados):
    validos = np.isfinite(valores)
    bins = np.linspace(valores[validos].min(), valores[validos].max(), 36)
    for k, nome in [(0, "False (0)"), (1, "True (1)")]:
        ax.hist(valores[validos & (y_treino == k)], bins=bins, alpha=0.56,
                color=CORES[k], label=f"Transported = {nome}",
                edgecolor="white", linewidth=0.35)
    ax.set(yscale="log", xlabel=rotulo, ylabel="Passageiros por faixa (escala log)",
           title=titulo + f"\nn = {validos.sum()}")
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Figura 6 — Cauda de FoodCourt antes, durante e depois do pré-processamento",
             fontsize=14)
salvar_figura(fig, 6)

# Checagens necessárias para preparar a entrada da rede, sem treinar modelos.
assert np.array_equal(imputador_num.statistics_,
                      treino_bruto[numericas_originais].median().to_numpy())
assert np.allclose(escalonador.data_min_, treino_log.min().to_numpy())
assert np.allclose(escalonador.data_max_, treino_log.max().to_numpy())
assert np.allclose(treino_num["TotalSpend"], treino_num[gastos].sum(axis=1))
assert np.allclose(teste_num["TotalSpend"], teste_num[gastos].sum(axis=1))
for j, cats in enumerate(encoder.categories_):
    assert set(cats) == set(treino_cat[:, j])
for X, y in [(X_treino_final, y_treino), (X_teste_final, y_teste)]:
    assert X.shape[0] == len(y)
    assert X.shape[1] == len(nomes_features)
    assert np.isfinite(X).all()
    assert X.min() >= -1 - 1e-12 and X.max() <= 1 + 1e-12
assert X_treino_final.shape[1] == X_teste_final.shape[1]
assert not set(["Transported", "PassengerId", "Cabin", "Name"]) & set(nomes_features)

checagens = pd.DataFrame([
    {"Conjunto": nome, "shape": str(X.shape), "NaN": int(np.isnan(X).sum()),
     "Infinitos": int(np.isinf(X).sum()), "Mínimo": float(X.min()), "Máximo": float(X.max())}
    for nome, X in [("Treino", X_treino_final), ("Teste", X_teste_final)]
])
print(checagens.to_string(index=False))
print("Ordem final das features:", nomes_features)

# Saídas reutilizáveis e índices para auditar/reproduzir a divisão exata.
np.savez_compressed(OUT / "dados_preprocessados.npz", X_train=X_treino_final,
                    X_test=X_teste_final, y_train=y_treino, y_test=y_teste,
                    train_indices=idx_treino, test_indices=idx_teste,
                    feature_names=np.asarray(nomes_features, dtype=str))
pd.DataFrame(X_treino_final, columns=nomes_features).to_csv(OUT / "X_treino.csv", index=False)
pd.DataFrame(X_teste_final, columns=nomes_features).to_csv(OUT / "X_teste.csv", index=False)
pd.DataFrame({"indice_original": idx_treino, "Transported": y_treino}).to_csv(
    OUT / "y_treino.csv", index=False)
pd.DataFrame({"indice_original": idx_teste, "Transported": y_teste}).to_csv(
    OUT / "y_teste.csv", index=False)
faltantes.to_csv(OUT / "faltantes.csv", index=False)
rij.to_csv(OUT / "razoes_separacao.csv", index=False)
mistura.to_csv(OUT / "taxas_mistura.csv", index=False)
np.savez_compressed(OUT / "datasets_sinteticos.npz",
                    X_s05=datasets_2d[0.5], X_s1=datasets_2d[1.0],
                    X_s2=datasets_2d[2.0], X_s4=datasets_2d[4.0], y_2d=y_2d,
                    X_I=X_I, y_I=y_I, X_II=X_II, y_II=y_II)
# --8<-- [end:bloco_12]

# --8<-- [start:bloco_13]
resumo_metricas = {
    "taxas_mistura": {str(s): float(taxas[s]) for s in escalas},
    "menor_rij_s1": rmin, "par_menor_rij": par_min, "menor_rij_s2": rmin / 2,
    "primeira_escala_amostral_nao_separavel": float(primeiro_s),
    "distancia_centros_dataset_I": dist_I, "distancia_centros_dataset_II": dist_II,
    "variancia_explicada_dataset_I": variancia_I.tolist(),
    "variancia_explicada_dataset_II": variancia_II.tolist(),
    "proporcao_transportados": float(y_real.mean()),
    "foodcourt_media_treino": food_media_treino,
    "foodcourt_mediana_treino": food_mediana_treino,
    "shape_treino": list(X_treino_final.shape), "shape_teste": list(X_teste_final.shape),
    "intervalo_treino": [float(X_treino_final.min()), float(X_treino_final.max())],
    "intervalo_teste": [float(X_teste_final.min()), float(X_teste_final.max())],
    "nan_treino": int(np.isnan(X_treino_final).sum()),
    "nan_teste": int(np.isnan(X_teste_final).sum()),
    "celulas_teste_clipping": int(clip_mask.sum()), "erros_regra_radial": erros_radiais,
    "erros_separador_radial_da_amostra": erros_separador_amostra,
    "dataset_sha256": hash_obtido, "versoes": versoes,
}
(OUT / "metricas.json").write_text(json.dumps(resumo_metricas, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
# --8<-- [end:bloco_13]
