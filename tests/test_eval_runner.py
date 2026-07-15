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
