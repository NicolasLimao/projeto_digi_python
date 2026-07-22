-- Passa a retornar o id do documento. As CTEs ja computavam d.id como doc_id e
-- o descartavam no SELECT final; agora ele sai junto.
--
-- Ranking intocado de proposito: mesmas CTEs, mesma formula, mesma ordenacao,
-- mesmo LIMIT. Assim, qualquer variacao no eval apos esta migracao e ruido de
-- geracao (temperature=0.5), nao efeito desta mudanca.
--
-- O ::bigint e explicito para nao depender de documents.id ser integer ou bigint.
--
-- Ordem de aplicacao: rodar depois da 003 e ANTES de publicar o codigo Python
-- que le o id. Nesta ordem a coluna extra e ignorada com seguranca pelo codigo
-- antigo, e o codigo novo nunca roda contra a funcao sem id.

CREATE OR REPLACE FUNCTION public.match_documents_hybrid(
  query_text text,
  query_embedding text,
  match_count integer DEFAULT 10,
  full_text_weight double precision DEFAULT 0.3,
  semantic_weight double precision DEFAULT 0.7
)
RETURNS TABLE(id bigint, content text, metadata jsonb, score double precision)
LANGUAGE plpgsql
AS $function$
DECLARE clean_embedding VECTOR(1536);
BEGIN
  clean_embedding := LTRIM(query_embedding, '=')::vector;
  RETURN QUERY
  WITH semantic AS (
    SELECT d.id AS doc_id, d.content AS doc_content, d.metadata AS doc_metadata,
      (1 - (d.embedding <=> clean_embedding))::float AS sem_score
    FROM documents d
  ),
  full_text AS (
    SELECT d.id AS doc_id,
      ts_rank(to_tsvector('portuguese', d.content), plainto_tsquery('portuguese', query_text))::float AS ft_score
    FROM documents d
    WHERE to_tsvector('portuguese', d.content) @@ plainto_tsquery('portuguese', query_text)
  ),
  combined AS (
    SELECT s.doc_id, s.doc_content, s.doc_metadata,
      (semantic_weight * s.sem_score) + (full_text_weight * COALESCE(f.ft_score, 0)) AS combined_score
    FROM semantic s LEFT JOIN full_text f ON s.doc_id = f.doc_id
  )
  SELECT doc_id::bigint AS id, doc_content AS content, doc_metadata AS metadata, combined_score AS score
  FROM combined ORDER BY combined_score DESC LIMIT match_count;
END;
$function$;
