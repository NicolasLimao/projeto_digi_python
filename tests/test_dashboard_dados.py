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


def test_buckets_paginados_segue_next_page(monkeypatch):
    paginas = [
        {"data": [{"start_time": 1, "results": []}], "has_more": True, "next_page": "p2"},
        {"data": [{"start_time": 2, "results": []}], "has_more": False},
    ]
    chamadas: list[dict] = []

    class _Resposta:
        def __init__(self, corpo):
            self._corpo = corpo

        def raise_for_status(self):
            pass

        def json(self):
            return self._corpo

    def _fake_get(url, params, headers, timeout):
        chamadas.append(dict(params))
        return _Resposta(paginas[len(chamadas) - 1])

    monkeypatch.setattr(dados.httpx, "get", _fake_get)
    buckets = dados._buckets_paginados("https://exemplo", "sk-admin", 0, 31)
    assert len(buckets) == 2
    assert len(chamadas) == 2
    assert "page" not in chamadas[0]
    assert chamadas[1]["page"] == "p2"


def test_buckets_paginados_respeita_teto_de_seguranca(monkeypatch):
    def _sempre_tem_mais(url, params, headers, timeout):
        class _Resposta:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{}], "has_more": True, "next_page": "x"}

        return _Resposta()

    monkeypatch.setattr(dados.httpx, "get", _sempre_tem_mais)
    buckets = dados._buckets_paginados("https://exemplo", "sk-admin", 0, 31)
    assert len(buckets) == 12  # máx. 12 páginas de segurança


def test_custos_openai_usa_limites_distintos_para_costs_e_usage(monkeypatch):
    urls_e_limites: list[tuple[str, int]] = []

    def _fake_buckets(url, admin_key, inicio, limite_pagina):
        urls_e_limites.append((url, limite_pagina))
        return []

    monkeypatch.setattr(dados, "_buckets_paginados", _fake_buckets)
    resultado = dados.custos_openai("sk-admin", dias=90)
    assert resultado == {"por_dia": [], "custo_total_usd": 0.0}
    limites = dict((url.rsplit("/", 1)[-1], limite) for url, limite in urls_e_limites)
    assert limites["costs"] == 90
    assert limites["completions"] == 31  # usage é limitado a 31 mesmo com dias=90


def test_ingerir_resposta_corpo_2xx_nao_json(monkeypatch):
    class _Resposta:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(dados.httpx, "post", lambda *a, **k: _Resposta())
    resultado = dados.ingerir_resposta("https://api", "token", "texto")
    assert resultado == {"chunks_created": 0}


def _relatorio_exemplo() -> str:
    resumo = {
        "total_interacoes": 384,
        "positivos": 209,
        "negativos": 14,
        "sem_feedback": 161,
        "taxa_aprovacao_pct": 94,
    }
    por_modo = [{"modo": "orientacao", "interacoes": 332, "positivos": 178, "negativos": 11}]
    por_canal = [{"canal": "dm", "interacoes": 383, "positivos": 209, "negativos": 14}]
    baseline = {"run": "2026-07-15-2357", "aprovados": 17, "total": 32}
    duvida = dados.Duvida("prod:1", "producao", "orientacao", "Como exportar?", "R ruim")
    return dados.gerar_relatorio_md(
        resumo, por_modo, por_canal, baseline, [duvida], dias=30, gerado_em="2026-07-16 10:00"
    )


def test_relatorio_md_contem_secoes_e_dados():
    texto = _relatorio_exemplo()
    assert "# Relatório Digi" in texto
    assert "2026-07-16 10:00" in texto
    assert "384" in texto and "94" in texto
    assert "orientacao" in texto and "dm" in texto
    assert "17/32" in texto
    assert "Como exportar?" in texto


def test_relatorio_md_sem_baseline_e_sem_duvidas():
    texto = dados.gerar_relatorio_md(
        {"total_interacoes": 0}, [], [], None, [], dias=7, gerado_em="2026-07-16 10:00"
    )
    assert "Nenhuma rodada de avaliação encontrada" in texto
    assert "Nenhuma dúvida pendente" in texto


def test_relatorio_md_com_delta_do_baseline():
    baseline = {"run": "r", "aprovados": 20, "total": 32, "delta": 3}
    texto = dados.gerar_relatorio_md(
        {"total_interacoes": 0}, [], [], baseline, [], dias=7, gerado_em="2026-07-16 10:00"
    )
    assert "20/32" in texto
    assert "(delta +3" in texto


def test_relatorio_md_com_erro_duvidas_e_honesto():
    duvida = dados.Duvida("prod:1", "producao", "orientacao", "Como exportar?", "R ruim")
    texto = dados.gerar_relatorio_md(
        {"total_interacoes": 0},
        [],
        [],
        None,
        [duvida],
        dias=7,
        gerado_em="2026-07-16 10:00",
        erro_duvidas="Falha ao carregar as fontes de dúvidas: RuntimeError",
    )
    assert "Não foi possível apurar as dúvidas pendentes nesta geração." in texto
    assert "Como exportar?" not in texto
    assert "Nenhuma dúvida pendente" not in texto


def test_relatorio_pdf_gera_bytes_legiveis():
    import fitz

    conteudo = dados.gerar_relatorio_pdf(_relatorio_exemplo())
    assert conteudo[:5] == b"%PDF-"
    with fitz.open(stream=conteudo, filetype="pdf") as doc:
        assert doc.page_count >= 1
        texto = "".join(page.get_text() for page in doc)
    assert "Digi" in texto and "384" in texto


class _StubSupabase:
    def __init__(self):
        self.inserido: dict | None = None
        self.tabela: str | None = None

    def table(self, nome: str):
        self.tabela = nome
        return self

    def insert(self, payload: dict):
        self.inserido = payload
        return self

    def execute(self):
        return self


def test_descartar_duvida_registra_sem_ingerir():
    stub = _StubSupabase()
    duvida = dados.Duvida("prod:1", "producao", "orientacao", "comob", "resposta do bot")
    dados.descartar_duvida(stub, duvida, dados.MOTIVOS_DESCARTE["invalida"])
    assert stub.tabela == dados.TABELA_RESPOSTAS
    assert stub.inserido is not None
    assert stub.inserido["chave"] == "prod:1"
    assert stub.inserido["ingerida"] is False
    assert stub.inserido["chunks_criados"] == 0
    assert "descartada" in stub.inserido["resposta_correta"]
    assert dados.MOTIVOS_DESCARTE["invalida"] in stub.inserido["resposta_correta"]


class _StubDocs:
    """Stub do cliente Supabase para a tabela documents (resolução/backup/update)."""

    def __init__(self, linhas: list[dict] | None = None):
        self.linhas = linhas or []
        self.filtros: list[tuple[str, str]] = []
        self._modo = "select"
        self._payload: dict | None = None
        self._id_alvo = None
        self.updates: list[dict] = []

    def table(self, nome: str):
        self.tabela = nome
        return self

    def select(self, *_):
        self._modo = "select"
        return self

    def update(self, payload: dict):
        self._modo = "update"
        self._payload = payload
        return self

    def eq(self, coluna: str, valor):
        if coluna == "id":
            self._id_alvo = valor
        else:
            self.filtros.append((coluna, str(valor)))
        return self

    def execute(self):
        if self._modo == "update":
            existe = any(linha.get("id") == self._id_alvo for linha in self.linhas)
            self.updates.append({"id": self._id_alvo, **(self._payload or {})})
            dados_ret = [{"id": self._id_alvo}] if existe else []
            self._id_alvo = None
            return type("R", (), {"data": dados_ret})()
        linhas = self.linhas
        for coluna, valor in self.filtros:
            chave = coluna.split("->>")[-1]
            linhas = [
                linha for linha in linhas if str((linha.get("metadata") or {}).get(chave)) == valor
            ]
        if self._id_alvo is not None:
            linhas = [linha for linha in linhas if linha.get("id") == self._id_alvo]
        self.filtros = []
        self._id_alvo = None
        return type("R", (), {"data": [dict(linha) for linha in linhas]})()


def test_chunks_suspeitos_achata_instaveis_e_deduplica():
    mapa = {
        "casos": [
            {
                "id": "neg-008",
                "indice": 0.4,
                "chunks": [
                    {
                        "ref": "discord-upload#37",
                        "data": "2026-05-28",
                        "trecho": "A",
                        "score_busca": 0.4,
                    },
                    {
                        "ref": "discord-upload#41",
                        "data": "2026-05-28",
                        "trecho": "B",
                        "score_busca": 0.3,
                    },
                ],
            },
            {
                "id": "pos-002",
                "indice": 0.0,
                "chunks": [{"ref": "x#1", "data": "2026-05-28", "trecho": "Z", "score_busca": 0.6}],
            },
            {
                "id": "neg-013",
                "indice": 0.4,
                "chunks": [
                    {
                        "ref": "discord-upload#37",
                        "data": "2026-05-28",
                        "trecho": "A",
                        "score_busca": 0.4,
                    }
                ],
            },
        ]
    }
    suspeitos = dados.chunks_suspeitos(mapa)
    trechos = [s["trecho"] for s in suspeitos]
    assert trechos == ["A", "B"]  # dedupe por trecho; pos-002 (estável) fora
    assert suspeitos[0]["caso_id"] == "neg-008"


def test_mapa_sem_ref_detecta_formato_antigo():
    antigos = [{"caso_id": "neg-008", "ref": "", "trecho": "A"}]
    novos = [{"caso_id": "neg-008", "ref": "discord-upload#37", "trecho": "A"}]
    misto = [{"ref": ""}, {"ref": "discord-upload#37"}]
    assert dados.mapa_sem_ref(antigos) is True
    assert dados.mapa_sem_ref(novos) is False
    assert dados.mapa_sem_ref(misto) is False
    assert dados.mapa_sem_ref([]) is False


def test_classificar_idade():
    assert dados.classificar_idade("2026-07-16T00:00:00Z") == "curado"
    assert dados.classificar_idade("2026-05-28T22:14:23.192Z") == "manual"
    assert dados.classificar_idade("2026-04-14") == "manual"
    assert dados.classificar_idade("data-invalida") == "manual"


def test_resolver_chunk_desempata_por_conteudo_normalizado():
    linhas = [
        {
            "id": 977,
            "content": "Agente   Inteligente\ncontratado",
            "metadata": {"fonte": "discord-upload", "chunk_index": 37},
        },
        {
            "id": 45,
            "content": "Outro chunk qualquer",
            "metadata": {"fonte": "discord-upload", "chunk_index": 37},
        },
    ]
    stub = _StubDocs(linhas)
    trecho = dados._trecho_normalizado("Agente   Inteligente\ncontratado")
    achados = dados.resolver_chunk(stub, "discord-upload#37", trecho)
    assert [a["id"] for a in achados] == [977]


def test_resolver_chunk_sem_match_retorna_vazio():
    stub = _StubDocs(
        [
            {
                "id": 1,
                "content": "nada a ver",
                "metadata": {"fonte": "discord-upload", "chunk_index": 37},
            }
        ]
    )
    assert dados.resolver_chunk(stub, "discord-upload#37", "trecho inexistente") == []


def test_montar_backup_preserva_campos_do_original():
    linha = {"id": 977, "content": "velho", "embedding": [0.1, 0.2], "metadata": {"fonte": "x"}}
    backup = dados.montar_backup(linha)
    assert backup["id"] == 977
    assert backup["content"] == "velho"
    assert backup["embedding"] == [0.1, 0.2]
    assert backup["metadata"] == {"fonte": "x"}
    assert "backup_em" in backup


def test_salvar_backup_faz_append_jsonl(tmp_path):
    caminho = tmp_path / "documents-2026-07-19.jsonl"
    dados.salvar_backup(caminho, {"id": 1, "content": "a"})
    dados.salvar_backup(caminho, {"id": 2, "content": "b"})
    linhas = caminho.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 2
    assert json.loads(linhas[1])["id"] == 2


def test_atualizar_chunk_envia_content_e_embedding():
    stub = _StubDocs([{"id": 977, "content": "velho", "metadata": {}}])
    dados.atualizar_chunk(stub, 977, "novo texto", [0.5, 0.6])
    assert stub.updates == [{"id": 977, "content": "novo texto", "embedding": [0.5, 0.6]}]


def test_resolver_chunk_casa_trecho_com_email_redigido():
    linhas = [
        {
            "id": 5,
            "content": "contate joao@ikatec.com.br no suporte",
            "metadata": {"fonte": "discord-upload", "chunk_index": 9},
        },
    ]
    stub = _StubDocs(linhas)
    # o mapa guarda o trecho JÁ com o email mascarado
    achados = dados.resolver_chunk(stub, "discord-upload#9", "contate [email] no suporte")
    assert [a["id"] for a in achados] == [5]


class _StubOpenAI:
    def __init__(self, vetor: list[float]):
        self._vetor = vetor
        self.chamada: dict | None = None
        self.embeddings = self

    def create(self, model: str, input: str):
        self.chamada = {"model": model, "input": input}
        dado = type("D", (), {"embedding": self._vetor})()
        return type("R", (), {"data": [dado]})()


def test_reembed_usa_modelo_correto_e_retorna_vetor():
    stub = _StubOpenAI([0.1, 0.2, 0.3])
    vetor = dados.reembed(stub, "texto novo")
    assert vetor == [0.1, 0.2, 0.3]
    assert stub.chamada == {"model": dados.EMBEDDING_MODEL, "input": "texto novo"}


def test_buscar_documento_retorna_linha_ou_none():
    stub = _StubDocs([{"id": 7, "content": "x", "embedding": [0.1], "metadata": {}}])
    assert dados.buscar_documento(stub, 7)["content"] == "x"
    assert dados.buscar_documento(_StubDocs([]), 99) is None


def test_atualizar_chunk_sem_linha_afetada_levanta():
    import pytest

    with pytest.raises(RuntimeError, match="nenhuma linha"):
        dados.atualizar_chunk(_StubDocs([]), 999, "novo", [0.1])
