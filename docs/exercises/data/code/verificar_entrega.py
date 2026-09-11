"""Verifica o contrato da entrega e, opcionalmente, reexecuta os cálculos.

Uso, da raiz do repositório:
    python docs/exercises/data/code/verificar_entrega.py --reexecutar
    mkdocs build --strict
    python docs/exercises/data/code/verificar_entrega.py --site site

Usa apenas a biblioteca padrão; não treina modelos nem sorteia novos dados.
"""

import argparse
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

EXERCISE = Path(__file__).resolve().parent.parent
REPO = EXERCISE.parents[2]
EXPECTED_URL = "https://gubscruz.github.io/ann-dl/exercises/data/"


def iguais(a, b, caminho="métricas"):
    # Tolerância só para arredondamento de ponto flutuante entre plataformas.
    if isinstance(a, dict):
        assert a.keys() == b.keys(), caminho
        for key in a:
            if key != "versoes":
                iguais(a[key], b[key], f"{caminho}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), caminho
        for j, (x, y) in enumerate(zip(a, b)):
            iguais(x, y, f"{caminho}[{j}]")
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (caminho, a, b)
    else:
        assert a == b, (caminho, a, b)


class Pagina(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = []
        self.canonical = None
        self.textos = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag == "img" and attrs.get("src"):
            self.images.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href")

    def handle_data(self, data):
        self.textos.append(data)


def verificar_fontes():
    texto = (EXERCISE / "index.md").read_text(encoding="utf-8")
    assert texto.startswith("---\nexercise: data\n")
    front = texto.split("---", 2)[1]
    assert re.search(r'^ai_use:\s*".+"$', front, flags=re.M)

    titulos = re.findall(r"^(#{2,3}) (.+)$", texto, flags=re.M)
    estrutura = [(nivel, titulo.split(" — ")[0]) for nivel, titulo in titulos]
    esperado = [
        ("##", "Exercício 1"), ("###", "A"), ("###", "B"), ("###", "C"),
        ("##", "Exercício 2"), ("###", "A"), ("###", "B"), ("###", "C"), ("###", "D"),
        ("##", "Exercício 3"), ("###", "A"), ("###", "B"), ("###", "C"), ("###", "D"),
        ("##", "Resumo dos resultados"),
    ]
    assert estrutura == esperado, estrutura
    resumo = texto.split("## Resumo dos resultados", 1)[1]
    assert "| # | Item | Seu valor |" in resumo
    linhas = re.findall(r"^\| (\d+) \| (.+?) \| (.+?) \|$", resumo, flags=re.M)
    assert [int(numero) for numero, _, _ in linhas] == list(range(1, 14))
    assert all(valor.strip() for _, _, valor in linhas)

    imagens = re.findall(r"!\[Figura (\d+).*?\]\((figures/[^)]+)\)", texto)
    assert [int(numero) for numero, _ in imagens] == list(range(1, 7))
    for _, nome in imagens:
        assert (EXERCISE / nome).is_file(), nome

    includes = re.findall(r'--8<-- "([^"]+)"', texto)
    assert len(includes) == 13
    for ref in includes:
        arquivo, bloco = ref.rsplit(":", 1)
        assert arquivo == "docs/exercises/data/code/analise.py"
        fonte = (REPO / arquivo).read_text(encoding="utf-8")
        assert f"# --8<-- [start:{bloco}]" in fonte
        assert f"# --8<-- [end:{bloco}]" in fonte
    for path in (EXERCISE / "code").glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("Estrutura: front matter, títulos, 13 snippets, 6 figuras e resumo de 13 linhas OK.")


def verificar_site(pasta):
    site = Path(pasta).resolve()
    pagina = site / "exercises/data/index.html"
    assert pagina.is_file(), pagina
    parser = Pagina()
    parser.feed(pagina.read_text(encoding="utf-8"))
    assert parser.canonical == EXPECTED_URL, parser.canonical
    figs = [src for src in parser.images if "figures/figura_" in src]
    assert len(figs) == 6, figs
    # Todos os recursos locais referenciados precisam existir no site construído.
    for ref in parser.links + parser.images:
        url = urlsplit(ref)
        if url.scheme or url.netloc or not url.path:
            continue
        relative = unquote(url.path)
        if relative.startswith("/ann-dl/"):
            dest = site / relative.removeprefix("/ann-dl/")
        elif relative.startswith("/"):
            dest = site / relative.lstrip("/")
        else:
            dest = pagina.parent / relative
        if dest.is_dir():
            dest = dest / "index.html"
        assert dest.is_file(), (ref, str(dest))
    visible = "".join(parser.textos)
    assert "--8<--" not in visible, "Snippet não expandido no site."
    assert "rng = np.random.default_rng(42)" in visible
    assert "Uso de IA declarado" in visible
    assert "Resumo dos resultados" in visible
    print("Site: URL canônica, código expandido, 6 figuras e recursos locais OK.")


def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument("--reexecutar", action="store_true")
    args_parser.add_argument("--site", type=Path)
    args = args_parser.parse_args()
    verificar_fontes()
    if args.reexecutar:
        arquivo = EXERCISE / "resultados/metricas.json"
        antes = json.loads(arquivo.read_text(encoding="utf-8"))
        processo = subprocess.run(
            [sys.executable, str(EXERCISE / "code/analise.py")], cwd=REPO,
            env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(REPO / ".cache/matplotlib")},
            text=True, capture_output=True,
        )
        if processo.returncode:
            print(processo.stdout)
            print(processo.stderr, file=sys.stderr)
            raise SystemExit(processo.returncode)
        depois = json.loads(arquivo.read_text(encoding="utf-8"))
        iguais(antes, depois)
        print("Reprodução: código executado; todas as métricas coincidem com a entrega versionada.")
    if args.site:
        verificar_site(args.site)


if __name__ == "__main__":
    main()
