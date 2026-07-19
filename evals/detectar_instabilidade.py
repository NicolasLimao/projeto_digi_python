"""Detector de instabilidade da base RAG do Digi.

Roda cada caso do dataset N vezes contra a API, julga cada resposta com a
mesma rubrica do eval e mede a divergência dos vereditos. Captura os chunks
da busca híbrida para apontar conteúdo conflitante por data (insumo da
Fase 2: reconciliação). 100% leitura — nada é escrito no banco.

Uso:
    python evals/detectar_instabilidade.py                  # 5x por caso
    python evals/detectar_instabilidade.py --repeticoes 3
    python evals/detectar_instabilidade.py --filter neg --limit 5
    python evals/detectar_instabilidade.py --dry-run
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))
import run_eval  # noqa: E402, F401 - reusa dataset, juiz, parsing e constantes do runner

DEFAULT_REPETICOES = 5
LIMIAR_CONFLITO_DIAS = 30
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CONCURRENCY = 3
RPC_BUSCA = "match_documents_hybrid"


@dataclass
class ResultadoCaso:
    case_id: str
    pergunta: str
    vereditos: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)


def indice_instabilidade(vereditos: list[str]) -> float:
    """1 - freq(majoritário)/N sobre vereditos válidos; erros de infra ficam fora."""
    validos = [v for v in vereditos if v in ("aprovado", "reprovado")]
    if not validos:
        return 0.0
    majoritario = Counter(validos).most_common(1)[0][1]
    return round(1 - majoritario / len(validos), 3)


def contar_erros(vereditos: list[str]) -> int:
    return sum(1 for v in vereditos if v not in ("aprovado", "reprovado"))


def conflito_potencial(
    chunks: list[dict[str, Any]], limiar_dias: int = LIMIAR_CONFLITO_DIAS
) -> bool:
    """True quando os chunks recuperados misturam datas com diferença > limiar."""
    datas: list[date] = []
    for chunk in chunks:
        try:
            datas.append(date.fromisoformat(str(chunk.get("data"))[:10]))
        except ValueError:
            continue
    if len(datas) < 2:
        return False
    return (max(datas) - min(datas)).days > limiar_dias


def montar_relatorio(
    resultados: list[ResultadoCaso], repeticoes: int, run_id: str
) -> tuple[str, dict[str, Any]]:
    """Relatório md (humano) + estrutura json (Fase 2), ordenados por instabilidade."""
    ordenados = sorted(resultados, key=lambda r: indice_instabilidade(r.vereditos), reverse=True)
    instaveis = [r for r in ordenados if indice_instabilidade(r.vereditos) > 0]
    com_conflito = [r for r in ordenados if conflito_potencial(r.chunks)]

    linhas = [
        f"# Instabilidade {run_id}",
        "",
        f"**{len(instaveis)} casos instáveis de {len(resultados)}** "
        f"({repeticoes} repetições por caso) · "
        f"{len(com_conflito)} com conflito potencial de datas",
        "",
    ]
    for resultado in ordenados:
        indice = indice_instabilidade(resultado.vereditos)
        if indice == 0:
            continue
        aviso = " ⚠️ CONFLITO DE DATAS" if conflito_potencial(resultado.chunks) else ""
        linhas += [
            f"## {resultado.case_id} — índice {indice}{aviso}",
            "",
            f"Pergunta: {resultado.pergunta}",
            f"Vereditos: {', '.join(resultado.vereditos)}",
        ]
        if resultado.scores:
            linhas.append(f"Score da API: {min(resultado.scores):.2f}-{max(resultado.scores):.2f}")
        if resultado.chunks:
            linhas.append("Chunks recuperados (id · data · trecho):")
            linhas += [
                f"- `{chunk.get('id')}` · {chunk.get('data') or 'sem data'} · "
                f"{chunk.get('trecho', '')}"
                for chunk in resultado.chunks
            ]
        linhas.append("")
    estaveis = [r.case_id for r in ordenados if indice_instabilidade(r.vereditos) == 0]
    linhas += [f"Estáveis ({len(estaveis)}): {', '.join(estaveis) or 'nenhum'}", ""]

    estrutura = {
        "run": run_id,
        "repeticoes": repeticoes,
        "casos": [
            {
                "id": r.case_id,
                "indice": indice_instabilidade(r.vereditos),
                "vereditos": r.vereditos,
                "erros": contar_erros(r.vereditos),
                "conflito_potencial": conflito_potencial(r.chunks),
                "chunks": r.chunks,
            }
            for r in ordenados
        ],
    }
    return "\n".join(linhas), estrutura
