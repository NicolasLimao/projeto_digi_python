"""Testes offline do runner de avaliação (sem rede, sem OpenAI)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parents[1] / "evals" / "run_eval.py"
)
run_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["run_eval"] = run_eval
_SPEC.loader.exec_module(run_eval)


def test_load_dataset_le_o_dataset_real():
    casos = run_eval.load_dataset(run_eval.DATASET_PATH)
    assert len(casos) >= 25
    assert all(caso.fatos_esperados for caso in casos)


def test_load_dataset_rejeita_ids_duplicados(tmp_path):
    linha = json.dumps(
        {
            "id": "x-001",
            "origem": "feedback_positivo",
            "modo": "orientacao",
            "pergunta": "p?",
            "fatos_esperados": ["f"],
        }
    )
    arquivo = tmp_path / "d.jsonl"
    arquivo.write_text(linha + "\n" + linha + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicado"):
        run_eval.load_dataset(arquivo)


def test_build_judge_prompt_inclui_rubrica():
    caso = run_eval.EvalCase(
        id="neg-001",
        origem="feedback_negativo",
        modo="orientacao",
        pergunta="Tem limite de arquivos?",
        fatos_esperados=["cita o limite por arquivo"],
        erros_proibidos=["afirmar que não há limite"],
    )
    prompt = run_eval.build_judge_prompt(caso, "resposta do bot")
    assert "cita o limite por arquivo" in prompt
    assert "afirmar que não há limite" in prompt
    assert "resposta do bot" in prompt


def test_parse_judge_response_feliz():
    veredito, motivo = run_eval.parse_judge_response(
        '{"veredito": "aprovado", "motivo": "cobre os fatos"}'
    )
    assert veredito == "aprovado"
    assert motivo == "cobre os fatos"


def test_parse_judge_response_invalida_vira_erro():
    veredito, motivo = run_eval.parse_judge_response("não sou json")
    assert veredito == "erro"
    assert "parseável" in motivo


def test_calcular_delta_sem_anteriores_retorna_none():
    resultados = [run_eval.CaseResult("neg-001", "aprovado", "ok")]
    assert run_eval._calcular_delta(resultados, None) is None


def test_calcular_delta_identifica_regressoes_e_correcoes():
    resultados = [
        run_eval.CaseResult("neg-001", "reprovado", "piorou"),
        run_eval.CaseResult("neg-002", "aprovado", "melhorou"),
        run_eval.CaseResult("neg-003", "aprovado", "manteve"),
    ]
    anteriores = {"neg-001": "aprovado", "neg-002": "reprovado", "neg-003": "aprovado"}
    regressoes, correcoes = run_eval._calcular_delta(resultados, anteriores)
    assert regressoes == ["neg-001"]
    assert correcoes == ["neg-002"]


def test_sanitizar_md_celula_escapa_pipe_e_remove_quebras():
    texto = "motivo com | pipe\ne quebra\r\nde linha"
    resultado = run_eval._sanitizar_md_celula(texto)
    assert "\n" not in resultado
    assert "\r" not in resultado
    assert "\\|" in resultado


def test_vereditos_anteriores_ignora_rodadas_parciais(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "REPORTS_DIR", tmp_path)
    (tmp_path / "2026-01-01-0000.json").write_text(
        json.dumps({"run": "a", "parcial": False, "vereditos": {"neg-001": "aprovado"}}),
        encoding="utf-8",
    )
    (tmp_path / "2026-01-02-0000.json").write_text(
        json.dumps({"run": "b", "parcial": True, "vereditos": {"neg-001": "reprovado"}}),
        encoding="utf-8",
    )
    anteriores = run_eval._vereditos_anteriores()
    assert anteriores == {"neg-001": "aprovado"}
