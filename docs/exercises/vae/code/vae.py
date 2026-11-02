"""Atividade VAE: treina os dois modelos e salva métricas e Figuras 1–6.

Execute da raiz do projeto: python docs/exercises/vae/code/vae.py
"""

import csv
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms

PASTA = Path(__file__).resolve().parents[1]
FIGURAS = PASTA / "figures"
RESULTADOS = PASTA / "resultados"
EPOCAS = 20
LOTE = 128
BETA = 1.0


# --8<-- [start:modelo]
class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, d)
        self.logvar = nn.Linear(256, d)
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparametrizar(self, mu, logvar):
        # logvar = log(sigma²), então sigma = exp(logvar / 2).
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrizar(mu, logvar)
        return self.decoder(z), mu, logvar


def perda_vae(x_hat, x, mu, logvar):
    # Somamos pixels e dimensões latentes e dividimos SÓ pelas imagens.
    n = x.size(0)
    reconstrucao = F.binary_cross_entropy(x_hat, x, reduction="sum") / n
    kl = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - 1 - logvar) / n
    return reconstrucao + BETA * kl, reconstrucao, kl
# --8<-- [end:modelo]


# --8<-- [start:dados]
def carregar_dados():
    train_full = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    test = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )
    train, val = random_split(
        train_full, [50_000, 10_000],
        generator=torch.Generator().manual_seed(42),
    )
    return train, val, test
# --8<-- [end:dados]


# --8<-- [start:treino]
def executar_epoca(modelo, loader, otimizador=None):
    treinando = otimizador is not None
    modelo.train(treinando)
    soma_rec, soma_kl, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(treinando):
        for imagens, _ in loader:
            x = imagens.flatten(start_dim=1)  # [lote, 784], pixels em [0, 1]
            x_hat, mu, logvar = modelo(x)
            total, rec, kl = perda_vae(x_hat, x, mu, logvar)
            if treinando:
                otimizador.zero_grad()
                total.backward()
                otimizador.step()
            # Ponderar pelo lote evita dar peso excessivo ao último lote.
            soma_rec += rec.item() * len(x)
            soma_kl += kl.item() * len(x)
            n += len(x)
    return {"reconstrucao": soma_rec / n, "kl": soma_kl / n,
            "total": (soma_rec + BETA * soma_kl) / n}


def avaliar(modelo, loader):
    # Uma amostra de z por imagem estima o termo de reconstrução do ELBO.
    # A semente fixa torna a avaliação repetível, sem mudar o RNG do treino.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return executar_epoca(modelo, loader)


def treinar(d, train, val):
    torch.manual_seed(42)  # Reiniciada ANTES de construir cada modelo.
    modelo = VAE(d)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    train_loader = DataLoader(
        train, batch_size=LOTE, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(val, batch_size=LOTE, shuffle=False, num_workers=0)
    historico = []
    for epoca in range(1, EPOCAS + 1):
        treino = executar_epoca(modelo, train_loader, otimizador)
        validacao = avaliar(modelo, val_loader)
        historico.append({"epoca": epoca,
                          "treino_rec": treino["reconstrucao"],
                          "treino_kl": treino["kl"],
                          "val_rec": validacao["reconstrucao"],
                          "val_kl": validacao["kl"]})
        print(f"d={d:2d} | época {epoca:2d}/{EPOCAS} | "
              f"treino: rec={treino['reconstrucao']:.2f}, KL={treino['kl']:.2f} | "
              f"val: rec={validacao['reconstrucao']:.2f}, KL={validacao['kl']:.2f}",
              flush=True)
    # Orçamento fixo: usamos a época 20, sem selecionar pelo teste.
    return modelo, historico
# --8<-- [end:treino]


def salvar_figura(fig, numero):
    fig.savefig(FIGURAS / f"figura_{numero}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def mosaico(imagens, linhas, colunas):
    """Junta as imagens 28×28 em uma única matriz para o imshow."""
    a = imagens.detach().cpu().numpy().reshape(linhas, colunas, 28, 28)
    return a.transpose(0, 2, 1, 3).reshape(linhas * 28, colunas * 28)


@torch.no_grad()
def gerar_figuras(modelo16, modelo2, historico16, test):
    modelo16.eval()
    modelo2.eval()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})

    # Figura 1: os dois termos separados, sempre em nats por imagem.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    epocas = [h["epoca"] for h in historico16]
    for ax, chave, titulo in zip(axes, ("rec", "kl"), ("Reconstrução (BCE)", "KL")):
        ax.plot(epocas, [h[f"treino_{chave}"] for h in historico16], label="Treino")
        ax.plot(epocas, [h[f"val_{chave}"] for h in historico16], label="Validação")
        ax.set(title=titulo, xlabel="Época", ylabel="Nats por imagem",
               xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Figura 1 — Treinamento do VAE com d = 16")
    salvar_figura(fig, 1)

    # Figura 2: primeira ocorrência de cada classe no teste, sem seleção visual.
    indices = [int((test.targets == digito).nonzero()[0]) for digito in range(10)]
    originais = torch.stack([test[i][0] for i in indices]).flatten(1)
    mu, _ = modelo16.encode(originais)
    reconstruidas = modelo16.decoder(mu)  # Aqui NÃO há sorteio de z.
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.imshow(mosaico(torch.cat([originais, reconstruidas]), 2, 10),
              cmap="gray", vmin=0, vmax=1)
    ax.set(xticks=np.arange(10) * 28 + 13.5, xticklabels=list(range(10)),
           yticks=[13.5, 41.5], yticklabels=["Original", "Reconstrução"],
           xlabel="Dígito", ylabel="Imagem",
           title="Figura 2 — Reconstruções a partir de μ (d = 16)")
    salvar_figura(fig, 2)

    # Figura 3: amostragem escrita explicitamente, com a semente solicitada.
    torch.manual_seed(0)
    z = torch.randn(64, 16)
    amostras = modelo16.decoder(z)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(mosaico(amostras, 8, 8), cmap="gray", vmin=0, vmax=1)
    centros = np.arange(8) * 28 + 13.5
    ax.set(xticks=centros, xticklabels=range(1, 9),
           yticks=centros, yticklabels=range(1, 9), xlabel="Coluna", ylabel="Linha",
           title="Figura 3 — 64 amostras do prior N(0, I), d = 16")
    salvar_figura(fig, 3)

    # Figura 4: todos os 10 mil pontos, usando a média da posterior.
    medias, rotulos = [], []
    for imagens, y in DataLoader(test, batch_size=LOTE, shuffle=False):
        mu, _ = modelo2.encode(imagens.flatten(1))
        medias.append(mu)
        rotulos.append(y)
    medias = torch.cat(medias).numpy()
    rotulos = torch.cat(rotulos).numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    for digito in range(10):
        pontos = medias[rotulos == digito]
        ax.scatter(pontos[:, 0], pontos[:, 1], s=6, alpha=0.5,
                   color=plt.get_cmap("tab10")(digito), label=str(digito))
    ax.set(xlabel="μ₁", ylabel="μ₂", title="Figura 4 — As 10.000 imagens de teste no latente 2-D")
    ax.legend(title="Dígito", bbox_to_anchor=(1.02, 1), loc="upper left",
              markerscale=3)
    ax.grid(alpha=0.15)
    salvar_figura(fig, 4)
    np.savez_compressed(RESULTADOS / "latente_2d.npz", mu=medias, rotulos=rotulos)

    # Figura 5: passos iguais em probabilidade, não em distância latente.
    q = np.linspace(0.05, 0.95, 15)
    valores = norm.ppf(q)
    # z2 decresce nas linhas: assim o eixo vertical cresce para cima.
    pontos = torch.tensor([(z1, z2) for z2 in valores[::-1] for z1 in valores],
                          dtype=torch.float32)
    plano = modelo2.decoder(pontos)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(mosaico(plano, 15, 15), cmap="gray", vmin=0, vmax=1)
    posicoes = np.arange(15) * 28 + 13.5
    ax.set(xticks=posicoes, xticklabels=[f"{v:.2f}" for v in valores],
           yticks=posicoes, yticklabels=[f"{v:.2f}" for v in valores[::-1]],
           xlabel="z₁ = Φ⁻¹(q₁)", ylabel="z₂ = Φ⁻¹(q₂)",
           title="Figura 5 — Plano latente decodificado (15 × 15)")
    ax.tick_params(axis="x", labelsize=8, labelrotation=60)
    ax.tick_params(axis="y", labelsize=8)
    salvar_figura(fig, 5)

    # Figura 6: os índices 0 e 1 têm classes diferentes (7 e 2).
    a, b = 0, 1
    xa, ya = test[a]
    xb, yb = test[b]
    assert ya != yb
    t = torch.linspace(0, 1, 10).unsqueeze(1)
    pixels = (1 - t) * xa.flatten() + t * xb.flatten()
    mu_ab, _ = modelo16.encode(torch.stack([xa.flatten(), xb.flatten()]))
    latentes = modelo16.decoder((1 - t) * mu_ab[0] + t * mu_ab[1])
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.imshow(mosaico(torch.cat([pixels, latentes]), 2, 10), cmap="gray", vmin=0, vmax=1)
    ax.set(xticks=np.arange(10) * 28 + 13.5,
           xticklabels=[f"{v:.2f}" for v in t.flatten().tolist()],
           yticks=[13.5, 41.5], yticklabels=["Pixels", "Latente (d=16)"],
           xlabel="t", ylabel="Interpolação",
           title=f"Figura 6 — Interpolação: teste[{a}] = {ya} → teste[{b}] = {yb}")
    salvar_figura(fig, 6)
    return {"primeiros_indices_por_digito": indices,
            "interpolacao_indices": [a, b], "interpolacao_digitos": [ya, yb],
            "quantis": q.tolist(), "coordenadas_grade": valores.tolist()}


def main():
    inicio = time.perf_counter()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    # CPU e número fixo de threads facilitam repetir esta execução.
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    np.random.seed(42)
    train, val, test = carregar_dados()
    modelos, historicos = {}, {}
    for d in (16, 2):
        modelos[d], historicos[d] = treinar(d, train, val)
        with (RESULTADOS / f"historico_d{d}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=historicos[d][0].keys())
            writer.writeheader()
            writer.writerows(historicos[d])
        # Pesos locais: úteis para conferir resultados, mas não vão para o site.
        Path("modelos").mkdir(exist_ok=True)
        torch.save(modelos[d].state_dict(), f"modelos/vae_d{d}.pt")

    # O teste só é avaliado DEPOIS de terminar os dois treinamentos.
    test_loader = DataLoader(test, batch_size=LOTE, shuffle=False)
    testes = {str(d): avaliar(modelos[d], test_loader) for d in (16, 2)}
    info_figuras = gerar_figuras(modelos[16], modelos[2], historicos[16], test)
    metricas = {
        "configuracao": {"semente": 42, "epocas": EPOCAS, "lote": LOTE,
                         "lr": 0.001, "beta": BETA, "device": "cpu", "threads": 4,
                         "n_treino": len(train), "n_validacao": len(val), "n_teste": len(test)},
        "versoes": {"python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__, "scipy": scipy.__version__},
        "teste": testes, "historicos": {str(d): historicos[d] for d in (16, 2)},
        "figuras": info_figuras, "tempo_segundos": time.perf_counter() - inicio,
    }
    (RESULTADOS / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nResultados finais (nats por imagem):", flush=True)
    print(json.dumps(testes, indent=2), flush=True)
    print(f"Tempo total: {metricas['tempo_segundos']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
