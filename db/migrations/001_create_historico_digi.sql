-- Create historico_digi table
CREATE TABLE historico_digi (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL,
  pergunta TEXT NOT NULL,
  resposta TEXT NOT NULL,
  modo TEXT DEFAULT 'orientacao',  -- orientacao | resposta-cliente | bug
  score FLOAT DEFAULT 0,
  chunks_used INT DEFAULT 0,
  processing_time_ms INT DEFAULT 0,
  timestamp TIMESTAMP DEFAULT NOW(),
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for fast user_id lookups
CREATE INDEX idx_historico_user_id ON historico_digi(user_id, timestamp DESC);
CREATE INDEX idx_historico_timestamp ON historico_digi(timestamp DESC);

-- Enable RLS (Row Level Security)
ALTER TABLE historico_digi ENABLE ROW LEVEL SECURITY;

-- Create policy for authenticated access (adjust based on your auth model)
CREATE POLICY "Allow service role full access" ON historico_digi
  USING (true) WITH CHECK (true);
