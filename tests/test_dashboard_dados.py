"""Testes offline da camada de dados do dashboard (sem rede, sem Streamlit)."""

import json
from datetime import date
from pathlib import Path

from dashboard import dados


def _escrever_jsonl(caminho: Path, casos: list[dict]) -> None:
    caminho.write_text(
        "\n".join(json.dumps(caso, ensure_ascii=False) for caso in casos) + "\n",
        encoding="utf-8",
    )


def test_duvidas_producao_converte_view():
    linhas = [
        {
            "timestamp": "2026-07-03T16:22:00+00:00",
            "canal": "dm",
            "modo": "orientacao",
            "score": 0.33,
            "pergunta": "P?",
            "resposta": "R ruim",
        }
    ]
    duvidas = dados.duvidas_producao(linhas)
    assert len(duvidas) == 1
    assert duvidas[0].chave == "prod:2026-07-03T16:22:00+00:00"
    assert duvidas[0].origem == "producao"
    assert duvidas[0].resposta_ruim == "R ruim"


def test_duvidas_eval_filtra_revisar_e_anota_veredito(tmp_path):
    _escrever_jsonl(
        tmp_path / "d.jsonl",
        [
            {
                "id": "neg-001",
                "origem": "feedback_negativo",
                "modo": "orientacao",
                "pergunta": "P1?",
                "fatos_esperados": ["f"],
                "resposta_anterior": "ruim",
                "notas": "REVISAR: confirmar",
            },
            {
                "id": "pos-001",
                "origem": "feedback_positivo",
                "modo": "orientacao",
                "pergunta": "P2?",
                "fatos_esperados": ["f"],
                "notas": "ok",
            },
        ],
    )
    duvidas = dados.duvidas_eval(tmp_path / "d.jsonl", {"neg-001": "reprovado"})
    assert [d.chave for d in duvidas] == ["eval:neg-001"]
    assert duvidas[0].veredito == "reprovado"


def test_pendentes_subtrai_respondidas_e_deduplica():
    a = dados.Duvida("prod:1", "producao", "orientacao", "P", "R")
    b = dados.Duvida("eval:x", "eval", "orientacao", "P", "R")
    duplicada = dados.Duvida("prod:1", "producao", "orientacao", "P", "R")
    resultado = dados.pendentes([a, duplicada], [b], {"eval:x"})
    assert [d.chave for d in resultado] == ["prod:1"]


def test_ultimo_baseline_ignora_parciais_e_calcula_delta(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps({"run": "r1", "vereditos": {"x": "aprovado", "y": "reprovado"}}),
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"run": "r2", "parcial": True, "vereditos": {"x": "reprovado"}}),
        encoding="utf-8",
    )
    (tmp_path / "c.json").write_text(
        json.dumps(
            {"run": "r3", "parcial": False, "vereditos": {"x": "aprovado", "y": "aprovado"}}
        ),
        encoding="utf-8",
    )
    baseline = dados.ultimo_baseline(tmp_path)
    assert baseline is not None
    assert (baseline["run"], baseline["aprovados"], baseline["total"]) == ("r3", 2, 2)
    assert baseline["delta"] == 1  # r3 (2 aprovados) vs r1 (1 aprovado); parcial ignorado
    assert baseline["vereditos"] == {"x": "aprovado", "y": "aprovado"}


def test_ultimo_baseline_sem_relatorios(tmp_path):
    assert dados.ultimo_baseline(tmp_path) is None


def test_janelas_volume_soma_janela_atual_e_anterior():
    linhas = [
        {"dia": "2026-07-15", "interacoes": 10, "positivos": 4, "negativos": 1},
        {"dia": "2026-07-10", "interacoes": 6, "positivos": 2, "negativos": 0},
        {"dia": "2026-07-05", "interacoes": 8, "positivos": 3, "negativos": 2},
    ]
    # janela atual (7d, 09-15/07): linhas de 15/07 e 10/07; anterior (02-08/07): linha de 05/07
    atual, anterior = dados.janelas_volume(linhas, dias=7, hoje=date(2026, 7, 15))
    assert atual == {"interacoes": 16, "positivos": 6, "negativos": 1}
    assert anterior == {"interacoes": 8, "positivos": 3, "negativos": 2}


def test_preparar_volume_grafico_formato_longo_com_sem_avaliacao():
    linhas = [{"dia": "2026-07-15", "interacoes": 10, "positivos": 4, "negativos": 1}]
    longo = dados.preparar_volume_grafico(linhas, dias=7, hoje=date(2026, 7, 15))
    categorias = {(item["categoria"], item["quantidade"]) for item in longo}
    assert categorias == {("Positivos", 4), ("Sem avaliação", 5), ("Negativos", 1)}


def test_texto_ingestao_estrutura():
    texto = dados.texto_ingestao("  Como faço X?  ", "  Passo a passo Y.  ")
    assert texto == (
        "Pergunta: Como faço X?\nResposta oficial validada pelo analista: Passo a passo Y."
    )


def test_parse_custos_agrega_buckets_diarios():
    costs = {
        "data": [
            {
                "start_time": 1784246400,  # 2026-07-17 UTC? valor exato não importa, ver dia
                "results": [{"amount": {"value": 0.12}}, {"amount": {"value": 0.03}}],
            }
        ]
    }
    usage = {
        "data": [
            {
                "start_time": 1784246400,
                "results": [
                    {"input_tokens": 1000, "output_tokens": 200},
                    {"input_tokens": 500, "output_tokens": 100},
                ],
            }
        ]
    }
    resultado = dados.parse_custos(costs, usage)
    assert len(resultado["por_dia"]) == 1
    dia = resultado["por_dia"][0]
    assert dia["custo_usd"] == 0.15
    assert dia["tokens_entrada"] == 1500
    assert dia["tokens_saida"] == 300
    assert resultado["custo_total_usd"] == 0.15
