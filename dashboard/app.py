"""Dashboard do Digi — rode com: streamlit run dashboard/app.py"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import altair as alt
import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import dados

st.set_page_config(page_title="Digi Dashboard", page_icon="🤖", layout="wide")

ENV = dados.carregar_env()
API_BASE = os.environ.get("DIGI_API_BASE") or ENV.get("DIGI_API_BASE") or dados.DEFAULT_API_BASE


@st.cache_resource(show_spinner=False)
def _cliente() -> Any:
    return dados.criar_cliente_supabase(ENV)


@st.cache_data(ttl=300, show_spinner="Consultando Supabase...")
def _view(nome: str) -> list[dict[str, Any]]:
    resposta = _cliente().table(nome).select("*").execute()
    return list(resposta.data or [])


def _limpar_caches() -> None:
    _view.clear()


st.title("🤖 Digi — Dashboard")
st.caption(f"API: {API_BASE} • dados do Supabase • atualizados a cada 5 min")

try:
    resumo_rows = _view("v_feedback_resumo")
except Exception:
    st.error(
        "Não consegui consultar o Supabase. Confira SUPABASE_URL e a chave no .env "
        "e sua conexão, depois recarregue."
    )
    st.stop()

resumo: dict[str, Any] = resumo_rows[0] if resumo_rows else {}
volume = _view("v_volume_diario")
por_modo = _view("v_por_modo")
por_canal = _view("v_por_canal")
baseline = dados.ultimo_baseline(dados.REPORTS_DIR)

# Dúvidas pendentes calculadas ANTES das abas: o export (aba 1) também as usa.
erro_duvidas: str | None = None
lista_pendentes: list[dados.Duvida] = []
producao: list[dados.Duvida] = []
eval_: list[dados.Duvida] = []
try:
    respondidas = dados.chaves_respondidas(_cliente())
    producao = dados.duvidas_producao(_view("v_negativos"))
    eval_ = dados.duvidas_eval(dados.DATASET_PATH, baseline["vereditos"] if baseline else None)
    lista_pendentes = dados.pendentes(producao, eval_, respondidas)
except Exception:
    erro_duvidas = (
        "Não consegui ler a tabela duvidas_respondidas. "
        "Rode sql/duvidas_respondidas.sql no Supabase SQL Editor."
    )
    respondidas = set()

aba_metricas, aba_duvidas = st.tabs(["📊 Métricas", "❓ Dúvidas pendentes"])

# ---------------------------------------------------------------- Métricas
with aba_metricas:
    dias = st.radio(
        "Período", [7, 30, 90], index=1, horizontal=True, format_func=lambda d: f"{d} dias"
    )
    hoje = datetime.now(UTC).date()
    atual, anterior = dados.janelas_volume(volume, dias, hoje)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Interações (janela)",
        atual["interacoes"],
        delta=atual["interacoes"] - anterior["interacoes"],
    )
    col2.metric(
        "Positivos (janela)",
        atual["positivos"],
        delta=atual["positivos"] - anterior["positivos"],
    )
    col3.metric(
        "Negativos (janela)",
        atual["negativos"],
        delta=atual["negativos"] - anterior["negativos"],
        delta_color="inverse",
    )
    col4.metric("Taxa de aprovação (geral)", f"{resumo.get('taxa_aprovacao_pct', 0)}%")
    if baseline:
        delta_eval = baseline.get("delta")
        col5.metric(
            "Eval (último baseline)",
            f"{baseline['aprovados']}/{baseline['total']}",
            delta=delta_eval if isinstance(delta_eval, int) else None,
        )
    else:
        col5.metric("Eval", "—")
        col5.caption("Rode `evals/run_eval.py` para gerar o baseline.")

    st.subheader("Volume diário")
    longo = dados.preparar_volume_grafico(volume, dias, hoje)
    if longo:
        grafico = (
            alt.Chart(pd.DataFrame(longo))
            .mark_bar(stroke="#1a1a19", strokeWidth=2)
            .encode(
                x=alt.X("dia:T", title=None),
                y=alt.Y("sum(quantidade):Q", title="Interações"),
                color=alt.Color(
                    "categoria:N",
                    scale=alt.Scale(
                        domain=["Positivos", "Sem avaliação", "Negativos"],
                        range=[
                            dados.COR_POSITIVO,
                            dados.COR_SEM_AVALIACAO,
                            dados.COR_NEGATIVO,
                        ],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                order=alt.Order("ordem:Q"),
                tooltip=[
                    alt.Tooltip("dia:T", title="Dia"),
                    alt.Tooltip("categoria:N", title="Categoria"),
                    alt.Tooltip("quantidade:Q", title="Quantidade"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(grafico, use_container_width=True)
    else:
        st.info("Sem interações no período selecionado.")

    col_modo, col_canal = st.columns(2)

    def _barras(titulo: str, linhas: list[dict[str, Any]], dimensao: str) -> Any:
        return (
            alt.Chart(pd.DataFrame(linhas))
            .mark_bar(color=dados.COR_SERIE, stroke="#1a1a19", strokeWidth=2)
            .encode(
                x=alt.X("interacoes:Q", title="Interações"),
                y=alt.Y(f"{dimensao}:N", sort="-x", title=None),
                tooltip=[dimensao, "interacoes", "positivos", "negativos"],
            )
            .properties(height=180, title=titulo)
        )

    with col_modo:
        if por_modo:
            st.altair_chart(_barras("Por modo", por_modo, "modo"), use_container_width=True)
            st.dataframe(pd.DataFrame(por_modo), use_container_width=True, hide_index=True)
    with col_canal:
        if por_canal:
            st.altair_chart(_barras("Por canal", por_canal, "canal"), use_container_width=True)
            st.dataframe(pd.DataFrame(por_canal), use_container_width=True, hide_index=True)

    st.subheader("Custos de IA (OpenAI)")
    chave_admin = ENV.get("OPENAI_ADMIN_KEY")
    if not chave_admin:
        st.info(
            "Para ver custo e tokens por dia, crie uma Admin API key em "
            "platform.openai.com → Settings → Organization → Admin keys e adicione "
            "`OPENAI_ADMIN_KEY=...` ao .env."
        )
    else:
        try:
            custos = st.session_state.get(f"custos_{dias}")
            if custos is None:
                custos = dados.custos_openai(chave_admin, dias)
                st.session_state[f"custos_{dias}"] = custos
        except httpx.HTTPError as erro:
            st.warning(f"Não consegui consultar a API de custos da OpenAI: {erro}")
        else:
            if custos and custos["por_dia"]:
                col_a, col_b = st.columns([1, 3])
                col_a.metric("Custo no período", f"US$ {custos['custo_total_usd']:.2f}")
                grafico_custos = (
                    alt.Chart(pd.DataFrame(custos["por_dia"]))
                    .mark_bar(color=dados.COR_SERIE, stroke="#1a1a19", strokeWidth=2)
                    .encode(
                        x=alt.X("dia:T", title=None),
                        y=alt.Y("custo_usd:Q", title="US$/dia"),
                        tooltip=["dia", "custo_usd", "tokens_entrada", "tokens_saida"],
                    )
                    .properties(height=200)
                )
                col_b.altair_chart(grafico_custos, use_container_width=True)
            else:
                st.info("Sem consumo registrado no período.")

    st.divider()
    st.subheader("Exportar relatório")
    relatorio_md = dados.gerar_relatorio_md(
        resumo,
        por_modo,
        por_canal,
        baseline,
        lista_pendentes,
        dias,
        datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
    formato = st.radio("Formato", ["TXT/Markdown", "PDF"], horizontal=True)
    carimbo = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    if formato == "TXT/Markdown":
        st.download_button(
            "⬇️ Baixar relatório (.md)",
            relatorio_md.encode("utf-8"),
            file_name=f"relatorio-digi-{carimbo}.md",
            mime="text/markdown",
        )
    else:
        st.download_button(
            "⬇️ Baixar relatório (.pdf)",
            dados.gerar_relatorio_pdf(relatorio_md),
            file_name=f"relatorio-digi-{carimbo}.pdf",
            mime="application/pdf",
        )

# --------------------------------------------------------- Dúvidas pendentes
with aba_duvidas:
    token = ENV.get("API_AUTH_TOKEN")
    if not token:
        st.error("API_AUTH_TOKEN ausente no .env — necessário para ensinar o bot.")
        st.stop()
    if erro_duvidas:
        st.error(erro_duvidas)
        st.stop()

    lista = lista_pendentes

    topo_a, topo_b = st.columns([4, 1])
    topo_a.caption(
        f"{len(lista)} pendentes • {len(producao)} negativos de produção • "
        f"{len(eval_)} dúvidas do eval • {len(respondidas)} já respondidas"
    )
    if topo_b.button("🔄 Recarregar"):
        _limpar_caches()
        st.rerun()

    if not lista:
        st.success("🎉 Nenhuma dúvida pendente — tudo respondido!")

    for duvida in lista:
        rotulo_origem = "🔴 produção" if duvida.origem == "producao" else "🧪 eval"
        with st.expander(f"{rotulo_origem} · {duvida.pergunta[:100]}"):
            st.markdown(f"**Pergunta completa:**\n\n> {duvida.pergunta}")
            if duvida.resposta_ruim:
                st.markdown("**Resposta que o bot deu (avaliada como ruim):**")
                st.code(duvida.resposta_ruim, language=None, wrap_lines=True)
            detalhes = [f"modo: {duvida.modo}"]
            if duvida.score is not None:
                detalhes.append(f"score: {duvida.score:.2f}")
            if duvida.veredito:
                detalhes.append(f"veredito do juiz: {duvida.veredito}")
            if duvida.timestamp:
                detalhes.append(f"em: {duvida.timestamp[:16]}")
            st.caption(" • ".join(detalhes))

            with st.form(key=f"form_{duvida.chave}"):
                resposta = st.text_area(
                    "Resposta correta (será ensinada ao bot)",
                    key=f"resp_{duvida.chave}",
                    height=160,
                    placeholder="Escreva a resposta oficial, como um analista experiente…",
                )
                enviado = st.form_submit_button("✅ Responder e ensinar ao bot")

            if enviado:
                if len(resposta.strip()) < 20:
                    st.error("Escreva uma resposta com pelo menos 20 caracteres.")
                else:
                    try:
                        resultado = dados.ingerir_resposta(
                            API_BASE,
                            token,
                            dados.texto_ingestao(duvida.pergunta, resposta),
                        )
                    except httpx.HTTPError as erro:
                        st.error(f"Falha na ingestão — a dúvida continua pendente. ({erro})")
                    else:
                        chunks = int(resultado.get("chunks_created") or 0)
                        try:
                            dados.registrar_resposta(_cliente(), duvida, resposta, chunks)
                        except Exception:
                            st.warning(
                                "A resposta FOI ingerida no RAG, mas falhou ao registrar "
                                "na tabela duvidas_respondidas. NÃO responda de novo — "
                                "insira o registro manualmente para não duplicar."
                            )
                        else:
                            st.toast(f"Bot ensinado! {chunks} chunks criados.", icon="🎓")
                            _limpar_caches()
                            st.rerun()
