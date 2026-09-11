# ann-dl · Redes Neurais e Deep Learning

Portfólio de **gubscruz** para a disciplina. O repositório mantém o histórico do
[template de hsandmann](https://github.com/hsandmann/documentation.template) e usa
MkDocs + Material, com publicação automática pelo GitHub Actions.

Endereço configurado para a entrega Data:
**https://gubscruz.github.io/ann-dl/exercises/data/**.
A disponibilidade pública depende da execução bem-sucedida do workflow Pages.

## Estrutura

```text
docs/
  index.md
  exercises/
    data/
      index.md                 # relatório com front matter e snippets
      code/
        analise.py             # código científico executável incluído na página
        gerar_relatorio.py     # recalcula números e reconstrói o relatório
        relatorio.ipynb        # notebook complementar executado
      data/train.csv           # dataset rotulado, conferido por SHA-256
      figures/figura_1.png ... figura_6.png
      resultados/              # métricas, matrizes e tabelas
    perceptron/index.md
    mlp/index.md
    vae/index.md
  projects/index.md
mkdocs.yml
requirements.txt
.github/workflows/pages.yml
```

As páginas Perceptron, MLP e VAE são espaços reservados para atividades futuras.
O relatório Data é a entrega completa atual. Os scripts estão em `code/`;
os blocos científicos exibidos no site vêm de `analise.py` por snippets
nomeados. As figuras são arquivos versionados, e a última seção é o resumo
com as 13 linhas do enunciado. O front matter e a página declaram o uso de IA.

## Reproduzir a entrega

Ambiente de referência: **Python 3.13.4**. Em um clone limpo:

```bash
git clone https://github.com/gubscruz/ann-dl.git
cd ann-dl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdocs build
```

Esses comandos compilam o site com o relatório e as figuras versionados.
Para reproduzir os cálculos, execute a análise separadamente, conforme abaixo.
Os cálculos usam apenas NumPy, pandas, Matplotlib e scikit-learn, este último
restrito à PCA e ao pré-processamento. Nenhum modelo é treinado.
As dependências MkDocs e Jupyter servem à apresentação do site e do notebook.

Para executar apenas a análise:

```bash
python docs/exercises/data/code/analise.py
```

Para recalcular também os números no texto e reconstruir o Markdown, o script
e o notebook após uma alteração:

```bash
python docs/exercises/data/code/gerar_relatorio.py
mkdocs build
```

Edite `gerar_relatorio.py` se precisar alterar parâmetros ou análises: ele é a
fonte que produz o relatório e `analise.py`. Alterações feitas diretamente nos
arquivos gerados podem ser sobrescritas.

Para visualizar localmente, execute `mkdocs serve` e abra o endereço informado.
As fórmulas usam MathJax por CDN; a renderização das fórmulas requer internet.
O notebook em `code/` também pode ser aberto com o kernel Python deste ambiente.

## Reprodutibilidade

Há apenas um `rng = np.random.default_rng(42)`. A ordem dos sorteios é:
nuvens originais; escalas 0,5, 2 e 4; gaussianas 5D por Cholesky; direções e
raios de C e D; split estratificado. O dataset de escala 1 reutiliza a nuvem
original e a PCA usa SVD completa. Alterar essa ordem altera os resultados
posteriores; executar apenas uma parte do notebook não reproduz a sequência.

As saídas incluem matrizes de treino `(6954, 21)` e teste `(1739, 21)`, sem NaN
ou infinitos, dentro de [−1, 1]. Imputadores, categorias one-hot e extremos da
escala são aprendidos apenas no treino. Índices da divisão estão preservados
em `resultados/dados_preprocessados.npz` e nos arquivos de rótulos.

## Fonte dos dados

[Kaggle — Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data),
Addison Howard, Ashley Chow e Ryan Holbrook (2022).
Somente o `train.csv` rotulado, com 8.693 linhas e 14 colunas, foi utilizado;
o teste do relatório é a divisão reservada desse arquivo.

Como o download direto do Kaggle exige autenticação, foi usada uma
[cópia pública](https://github.com/You-sha/Spaceship-Titanic/blob/main/train.csv),
conferida contra uma
[segunda cópia pública](https://github.com/AmirFARES/Kaggle-Spaceship-Titanic/blob/main/data/train.csv).
Ambas têm o mesmo SHA-256:

```text
17336d553f49ebdf6ecb266d2b5d3746e5dd308445f7c7864141c4f28d2a88d0
```

O CSV acompanha o repositório: não é necessário baixá-lo para reproduzir.
A igualdade dos espelhos não equivale à verificação criptográfica contra
um download autenticado do Kaggle; a origem é declarada no relatório.

## Publicação

O repositório deve permanecer **público**, com o nome **ann-dl**, na conta
**gubscruz**. Em **Settings → Pages**, a origem deve ser **GitHub Actions**.
O workflow `pages.yml` instala as dependências, compila com `mkdocs build`
e publica o conteúdo de `site/`. Pull requests somente compilam;
pushes em `main` e execução manual também publicam. O deploy não reexecuta
a análise nem confere critérios acadêmicos da entrega.

A URL canônica do relatório está configurada como
`https://gubscruz.github.io/ann-dl/exercises/data/`.
A data válida para a entrega é a do último commit em `docs/exercises/data/`;
a data da publicação não substitui esse registro. O histórico original do
template foi preservado e as alterações desta entrega têm commits próprios.
