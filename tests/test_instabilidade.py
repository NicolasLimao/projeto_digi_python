"""Testes offline do detector de instabilidade (lógica pura, sem rede)."""

import importlib.util
import sys
from pathlib import Path

_EVALS = Path(__file__).resolve().parents[1] / "evals"


def _carregar(nome: str):
    modulo_existente = sys.modules.get(nome)
    if modulo_existente is not None:
        return modulo_existente
    spec = importlib.util.spec_from_file_location(nome, _EVALS / f"{nome}.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


_carregar("run_eval")  # o detector faz `import run_eval`; registrar antes
det = _carregar("detectar_instabilidade")


def test_indice_zero_quando_unanime():
    assert det.indice_instabilidade(["aprovado"] * 5) == 0.0
    assert det.indice_instabilidade(["reprovado"] * 3) == 0.0


def test_indice_mede_divergencia():
    vereditos = ["aprovado", "reprovado", "aprovado", "reprovado", "aprovado"]
    assert det.indice_instabilidade(vereditos) == 0.4


def test_indice_ignora_erros_de_infra():
    assert det.indice_instabilidade(["aprovado", "erro", "erro", "aprovado"]) == 0.0
    assert det.indice_instabilidade(["aprovado", "reprovado", "erro"]) == 0.5
    assert det.indice_instabilidade(["erro", "erro"]) == 0.0


def test_contar_erros():
    assert det.contar_erros(["aprovado", "erro", "reprovado", "erro"]) == 2
    assert det.contar_erros(["aprovado"]) == 0


def test_conflito_potencial_por_datas():
    antigos_e_novos = [{"data": "2026-04-14"}, {"data": "2026-07-16"}]
    proximos = [{"data": "2026-05-01"}, {"data": "2026-05-20"}]
    sem_data = [{"data": None}, {"trecho": "x"}]
    um_so = [{"data": "2026-05-01"}]
    assert det.conflito_potencial(antigos_e_novos) is True
    assert det.conflito_potencial(proximos) is False
    assert det.conflito_potencial(sem_data) is False
    assert det.conflito_potencial(um_so) is False


def test_redigir_mascara_email_e_preserva_texto_normal():
    assert det._redigir("contate fulano.tal@exemplo.com para suporte") == (
        "contate [email] para suporte"
    )
    assert det._redigir("texto sem nenhum e-mail aqui") == "texto sem nenhum e-mail aqui"


def _resultados_sinteticos():
    instavel = det.ResultadoCaso(
        case_id="neg-008",
        pergunta="Quais IAs a plataforma digisac possui?",
        vereditos=["aprovado", "reprovado", "aprovado", "reprovado", "aprovado"],
        scores=[0.31, 0.38, 0.35, 0.33, 0.36],
        chunks=[
            {
                "ref": "discord-upload#1",
                "data": "2026-04-14",
                "trecho": "manual antigo",
                "score_busca": 0.4,
            },
            {
                "ref": "discord-upload#2",
                "data": "2026-07-16",
                "trecho": "material novo",
                "score_busca": 0.5,
            },
        ],
    )
    estavel = det.ResultadoCaso(
        case_id="pos-002",
        pergunta="o que são tags?",
        vereditos=["aprovado"] * 5,
        scores=[0.4] * 5,
        chunks=[
            {"ref": "discord-upload#3", "data": "2026-05-28", "trecho": "tags", "score_busca": 0.6}
        ],
    )
    return [estavel, instavel]


def test_relatorio_md_destaca_instaveis_e_conflito():
    md, _ = det.montar_relatorio(_resultados_sinteticos(), repeticoes=5, run_id="teste")
    assert "1 casos instáveis de 2" in md
    assert "neg-008" in md and "0.4" in md
    assert "CONFLITO DE DATAS" in md
    assert "pos-002" not in md.split("Estáveis")[0]  # estável não ganha seção própria
    assert "pos-002" in md.split("Estáveis")[1]


def test_relatorio_json_ordenado_e_sem_chave_vereditos_no_topo():
    _, estrutura = det.montar_relatorio(_resultados_sinteticos(), repeticoes=5, run_id="teste")
    assert "vereditos" not in estrutura  # invariante: não confundir o delta do run_eval
    assert estrutura["repeticoes"] == 5
    ids = [caso["id"] for caso in estrutura["casos"]]
    assert ids == ["neg-008", "pos-002"]  # mais instável primeiro
    assert estrutura["casos"][0]["conflito_potencial"] is True
    assert estrutura["casos"][0]["erros"] == 0
