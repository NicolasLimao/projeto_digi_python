-- Views de analytics do Digi (rode no Supabase SQL Editor)
-- Filtro "user_id ~ '^[0-9]+$'" mantém só usuários reais do Discord (IDs numéricos)
-- e exclui automaticamente user_ids de teste (teste_, base_, etc.)

-- 1. Resumo geral de feedback
create or replace view public.v_feedback_resumo as
select
  count(*)                                                   as total_interacoes,
  count(*) filter (where feedback = 'positivo')              as positivos,
  count(*) filter (where feedback = 'negativo')              as negativos,
  count(*) filter (where feedback is null)                   as sem_feedback,
  round(
    100.0 * count(*) filter (where feedback = 'positivo')
    / nullif(count(*) filter (where feedback in ('positivo','negativo')), 0)
  , 0)                                                       as taxa_aprovacao_pct
from public.historico_digi
where user_id ~ '^[0-9]+$';

-- 2. Volume e feedback por dia
create or replace view public.v_volume_diario as
select
  date(timestamp)                                as dia,
  count(*)                                        as interacoes,
  count(*) filter (where feedback = 'positivo')   as positivos,
  count(*) filter (where feedback = 'negativo')   as negativos
from public.historico_digi
where user_id ~ '^[0-9]+$'
group by date(timestamp)
order by dia desc;

-- 3. Negativos (acionáveis) — o que o bot errou/decepcionou
create or replace view public.v_negativos as
select
  timestamp, canal, modo, score,
  pergunta, pergunta_reescrita, resposta, fontes
from public.historico_digi
where user_id ~ '^[0-9]+$' and feedback = 'negativo'
order by timestamp desc;

-- 4. Desempenho por canal
create or replace view public.v_por_canal as
select
  coalesce(canal, 'desconhecido')                 as canal,
  count(*)                                        as interacoes,
  count(*) filter (where feedback = 'positivo')   as positivos,
  count(*) filter (where feedback = 'negativo')   as negativos
from public.historico_digi
where user_id ~ '^[0-9]+$'
group by coalesce(canal, 'desconhecido')
order by interacoes desc;

-- 5. Desempenho por modo (orientacao / resposta-cliente / bug)
create or replace view public.v_por_modo as
select
  modo,
  count(*)                                        as interacoes,
  count(*) filter (where feedback = 'positivo')   as positivos,
  count(*) filter (where feedback = 'negativo')   as negativos,
  round(avg(score)::numeric, 3)                   as score_medio
from public.historico_digi
where user_id ~ '^[0-9]+$'
group by modo
order by interacoes desc;
