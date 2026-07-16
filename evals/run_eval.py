"""Runner manual de avaliação RAG do Digi.

Roda os casos de evals/dataset.jsonl contra a API (produção por padrão) e
julga cada resposta com gpt-4o-mini usando a rubrica do caso. Gera relatório
datado em evals/reports/ (md + json) com delta vs rodada anterior.
Custo por rodada completa: centavos. NUNCA roda em CI.

Uso:
    python evals/run_eval.py                # rodada completa
    python evals/run_eval.py --filter neg   # só ids contendo "neg"
    python evals/run_eval.py --limit 5      # primeiros 5 casos
    python evals/run_eval.py --dry-run      # lista casos sem chamar a API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import dotenv_values
from openai import AsyncOpenAI

EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "dataset.jsonl"
REPORTS_DIR = EVALS_DIR / "reports"
DEFAULT_API_URL = "https://digi-api.squareweb.app/api/rag/query"
JUDGE_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 90.0
MAX_CONCURRENCY = 3

JUDGE_SYSTEM = (
    "Você é um avaliador rigoroso de respostas de um assistente de suporte da "
    "plataforma Digisac. Julgue apenas com base na rubrica fornecida. Responda "
    'somente JSON: {"veredito": "aprovado" ou "reprovado", "motivo": "<uma linha>"}'
)


@dataclass
class EvalCase:
    id: str
    origem: str
    modo: str
    pergunta: str
    fatos_esperados: list[str]
    erros_proibidos: list[str] = field(default_factory=list)
    resposta_anterior: str | None = None
    notas: str | None = None


@dataclass
class CaseResult:
    case_id: str
    veredito: str  # aprovado | reprovado | erro
    motivo: str
    score: float | None = None
    tempo_ms: float | None = None


def load_dataset(path: Path) -> list[EvalCase]:
    casos: list[EvalCase] = []
    for numero, linha in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not linha.strip():
            continue
        try:
            casos.append(EvalCase(**json.loads(linha)))
        except TypeError as exc:
            raise ValueError(f"caso inválido na linha {numero}: {exc}") from exc
    ids = [caso.id for caso in casos]
    if len(ids) != len(set(ids)):
        raise ValueError("id duplicado no dataset")
    return casos


def build_judge_prompt(caso: EvalCase, resposta: str) -> str:
    fatos = "\n".join(f"- {fato}" for fato in caso.fatos_esperados)
    partes = [
        f"PERGUNTA DO USUÁRIO:\n{caso.pergunta}",
        f"FATOS ESPERADOS (a resposta deve cobrir a essência de cada um):\n{fatos}",
    ]
    if caso.erros_proibidos:
        erros = "\n".join(f"- {erro}" for erro in caso.erros_proibidos)
        partes.append(f"ERROS PROIBIDOS (qualquer ocorrência reprova):\n{erros}")
    partes.append(f"RESPOSTA A JULGAR:\n{resposta}")
    return "\n\n".join(partes)


def parse_judge_response(texto: str) -> tuple[str, str]:
    try:
        dados = json.loads(texto)
        veredito = dados["veredito"]
        if veredito not in ("aprovado", "reprovado"):
            raise ValueError(veredito)
        return veredito, str(dados.get("motivo", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return "erro", f"resposta do juiz não parseável: {texto[:100]}"


async def _rodar_caso(
    caso: EvalCase,
    http_client: httpx.AsyncClient,
    juiz: AsyncOpenAI,
    api_url: str,
    api_token: str,
    user_id: str,
    limite: asyncio.Semaphore,
) -> CaseResult:
    async with limite:
        try:
            resposta_http = await http_client.post(
                api_url,
                params={"user_id": user_id, "canal": "eval"},
                json={"query": caso.pergunta, "mode": caso.modo},
                headers={"X-API-Key": api_token},
            )
            resposta_http.raise_for_status()
            corpo = resposta_http.json()
        except httpx.HTTPError as exc:
            return CaseResult(caso.id, "erro", f"falha na API: {type(exc).__name__}: {exc}")

        try:
            julgamento = await juiz.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": build_judge_prompt(caso, corpo.get("response", "")),
                    },
                ],
            )
        except Exception as exc:  # juiz indisponível não derruba a rodada
            return CaseResult(caso.id, "erro", f"falha no juiz: {type(exc).__name__}")

        veredito, motivo = parse_judge_response(julgamento.choices[0].message.content or "")
        return CaseResult(
            caso.id,
            veredito,
            motivo,
            score=corpo.get("score"),
            tempo_ms=corpo.get("processing_time_ms"),
        )


def _vereditos_anteriores() -> dict[str, str] | None:
    """Vereditos da última rodada *completa* (ignora rodadas parciais)."""
    for arquivo in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        if dados.get("parcial"):
            continue
        vereditos = dados.get("vereditos")
        if isinstance(vereditos, dict):
            return vereditos
    return None


def _calcular_delta(
    resultados: list[CaseResult], anteriores: dict[str, str] | None
) -> tuple[list[str], list[str]] | None:
    if not anteriores:
        return None
    regressoes = [
        r.case_id
        for r in resultados
        if r.veredito == "reprovado" and anteriores.get(r.case_id) == "aprovado"
    ]
    correcoes = [
        r.case_id
        for r in resultados
        if r.veredito == "aprovado" and anteriores.get(r.case_id) == "reprovado"
    ]
    return regressoes, correcoes


def _sanitizar_md_celula(texto: str) -> str:
    """Evita que `|` ou quebras de linha no texto do juiz quebrem a tabela md."""
    return texto.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _escrever_relatorio(
    resultados: list[CaseResult],
    run_id: str,
    anteriores: dict[str, str] | None,
    parcial: bool,
) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    aprovados = sum(1 for r in resultados if r.veredito == "aprovado")
    erros = sum(1 for r in resultados if r.veredito == "erro")

    linhas = [
        f"# Eval {run_id}",
        "",
        f"**Resultado: {aprovados}/{len(resultados)} aprovados** ({erros} erros)",
        "",
    ]
    delta = _calcular_delta(resultados, anteriores)
    if delta is not None:
        regressoes, correcoes = delta
        linhas += [
            f"Delta vs rodada anterior — regressões: {regressoes or 'nenhuma'}; "
            f"correções: {correcoes or 'nenhuma'}",
            "",
        ]
    linhas += ["| id | veredito | motivo | score | tempo (ms) |", "|---|---|---|---|---|"]
    for r in resultados:
        motivo = _sanitizar_md_celula(r.motivo)
        linhas.append(f"| {r.case_id} | {r.veredito} | {motivo} | {r.score} | {r.tempo_ms} |")

    md_path = REPORTS_DIR / f"{run_id}.md"
    md_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    (REPORTS_DIR / f"{run_id}.json").write_text(
        json.dumps(
            {
                "run": run_id,
                "parcial": parcial,
                "vereditos": {r.case_id: r.veredito for r in resultados},
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return md_path


async def _main_async(args: argparse.Namespace) -> int:
    todos_casos = load_dataset(DATASET_PATH)
    casos = todos_casos
    if args.filter:
        casos = [caso for caso in casos if args.filter in caso.id]
    if args.limit is not None:
        casos = casos[: args.limit]
    if not casos:
        print("Nenhum caso após filtros.")
        return 1
    parcial = bool(args.filter) or args.limit is not None or len(casos) < len(todos_casos)

    if args.dry_run:
        for caso in casos:
            print(f"{caso.id} [{caso.modo}] {caso.pergunta[:80]}")
        print(f"\n{len(casos)} casos (dry-run — nada foi chamado).")
        return 0

    env = dotenv_values(EVALS_DIR.parent / ".env")
    api_token = env.get("API_AUTH_TOKEN") or ""
    openai_key = env.get("OPENAI_API_KEY") or ""
    if not api_token or not openai_key:
        print("API_AUTH_TOKEN e OPENAI_API_KEY são obrigatórios no .env")
        return 1
    api_url = (
        args.api_url or env.get("EVAL_API_URL") or os.environ.get("EVAL_API_URL") or DEFAULT_API_URL
    )

    run_id = time.strftime("%Y-%m-%d-%H%M")
    user_id = f"eval_{time.strftime('%Y%m%d%H%M%S')}"
    anteriores = _vereditos_anteriores()
    limite = asyncio.Semaphore(MAX_CONCURRENCY)
    juiz = AsyncOpenAI(api_key=openai_key)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as http_client:
        resultados = list(
            await asyncio.gather(
                *(
                    _rodar_caso(caso, http_client, juiz, api_url, api_token, user_id, limite)
                    for caso in casos
                )
            )
        )
    await juiz.close()

    resultados.sort(key=lambda r: r.case_id)
    md_path = _escrever_relatorio(resultados, run_id, anteriores, parcial)

    aprovados = sum(1 for r in resultados if r.veredito == "aprovado")
    print(f"\n{aprovados}/{len(resultados)} aprovados — relatório: {md_path}")
    delta = _calcular_delta(resultados, anteriores)
    if delta is not None:
        regressoes, correcoes = delta
        print(
            f"Delta vs rodada anterior — regressões: {regressoes or 'nenhuma'}; "
            f"correções: {correcoes or 'nenhuma'}"
        )
    for r in resultados:
        if r.veredito != "aprovado":
            print(f"  {r.veredito.upper()}: {r.case_id} — {r.motivo}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Runner manual de avaliação RAG do Digi")
    parser.add_argument("--filter", help="roda só casos cujo id contém a substring")
    parser.add_argument("--limit", type=int, help="roda só os N primeiros casos")
    parser.add_argument("--dry-run", action="store_true", help="lista casos sem chamar a API")
    parser.add_argument(
        "--api-url",
        default=None,
        help="endpoint /api/rag/query (padrão: EVAL_API_URL do .env/ambiente, senão produção)",
    )
    return asyncio.run(_main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
