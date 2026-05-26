const { Client, GatewayIntentBits, ChannelType, Partials } = require('discord.js');
const axios = require('axios');
require('dotenv').config();

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.DirectMessages,
    GatewayIntentBits.MessageContent,
    GatewayIntentBits.GuildMessageReactions,
    GatewayIntentBits.DirectMessageReactions
  ],
  partials: [Partials.Channel, Partials.Message, Partials.Reaction]
});

// URLs e IDs
const N8N_WEBHOOK_INGESTAO = process.env.N8N_WEBHOOK_INGESTAO || 'http://localhost:5678/webhook/digi-ingestao';
const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000/api/rag/query';
const FEEDBACK_API_URL = RAG_API_URL.replace('/query', '/feedback');

const CANAL_INGESTAO = '1491637301522989198';
const CANAL_CONSULTA = '1491637352513142914';

// Mapeia: id da mensagem de resposta do bot -> interaction_id (para gravar feedback)
const feedbackMap = new Map();

console.log('[Bot] Iniciando...');
console.log('[Bot] RAG API URL:', RAG_API_URL);
console.log('[Bot] Feedback API URL:', FEEDBACK_API_URL);
console.log('[Bot] N8N Ingestão URL:', N8N_WEBHOOK_INGESTAO);

// Event: Bot conecta
client.on('ready', () => {
  console.log(`[Bot] ✅ Conectado como ${client.user.tag}`);
  console.log('[Bot] Escutando mensagens...');
});

// Quebra um texto longo em pedaços de até 1900 chars (limite do Discord)
function dividirMensagem(texto) {
  if (texto.length <= 1900) return [texto];
  const partes = [];
  let atual = '';
  texto.split('\n').forEach((linha) => {
    if ((atual + linha).length > 1900) {
      partes.push(atual);
      atual = linha;
    } else {
      atual += (atual ? '\n' : '') + linha;
    }
  });
  if (atual) partes.push(atual);
  return partes;
}

// Consulta o RAG, responde, e adiciona reações de feedback
async function handleRagQuery(message, userId, query, canal, logPrefix) {
  await message.channel.sendTyping();

  try {
    const url = new URL(RAG_API_URL);
    url.searchParams.append('user_id', userId);
    url.searchParams.append('canal', canal);

    const response = await axios.post(url.toString(), { query }, { timeout: 60000 });

    const result = response.data;
    const responseText = result.response || 'Sem resposta';
    const score = (result.score || 0).toFixed(2);
    const chunks = result.chunks_used || 0;
    const time = result.processing_time_ms || 0;

    console.log(`${logPrefix} ✅ Resposta (score=${score}, chunks=${chunks}, time=${time}ms)`);

    // Enviar resposta (dividindo se necessário) e guardar a última mensagem enviada
    const partes = dividirMensagem(responseText);
    let sentMsg;
    for (const parte of partes) {
      sentMsg = await message.reply(parte);
    }
    console.log(`${logPrefix} Enviada(s) ${partes.length} mensagem(ns)`);

    // Reações de feedback na resposta
    if (sentMsg && result.interaction_id) {
      try {
        await sentMsg.react('✅');
        await sentMsg.react('❌');
        feedbackMap.set(sentMsg.id, result.interaction_id);
      } catch (e) {
        console.error(`${logPrefix} Erro ao adicionar reações: ${e.message}`);
      }
    }

  } catch (error) {
    console.error(`${logPrefix} ❌ Erro: ${error.message}`);

    if (error.message.includes('ECONNREFUSED')) {
      await message.reply('❌ Servidor RAG offline. Tente novamente em alguns segundos.');
    } else if (error.code === 'ENOTFOUND') {
      await message.reply('❌ Erro de conexão com o servidor RAG.');
    } else if (error.code === 'ETIMEDOUT') {
      await message.reply('⏱️ Pergunta demorou muito para processar.');
    } else {
      await message.reply('❌ Erro ao processar pergunta. Tente novamente.');
    }
  }
}

// Event: Mensagem recebida
client.on('messageCreate', async (message) => {
  // Ignorar mensagens do bot
  if (message.author.bot) return;

  try {
    // 1. CANAL DE INGESTÃO (Feed de documentos) - mantém integridade com n8n
    if (message.channelId === CANAL_INGESTAO) {
      console.log(`[Ingestao] Documento de ${message.author.username}`);

      const payload = {
        channelId: message.channelId,
        content: message.content,
        id: message.id,
        attachments: message.attachments.map(a => ({
          url: a.url,
          filename: a.name,
          contentType: a.contentType
        }))
      };

      try {
        await axios.post(N8N_WEBHOOK_INGESTAO, payload, { timeout: 5000 });
        console.log(`[Ingestao] ✅ Enviado para n8n`);
      } catch (error) {
        console.error(`[Ingestao] ❌ Erro: ${error.message}`);
      }
      return;
    }

    // 2. CANAL DE CONSULTA (Dúvidas) - Python RAG
    if (message.channelId === CANAL_CONSULTA) {
      console.log(`[Consulta] ${message.author.username}: ${message.content.substring(0, 50)}...`);
      await handleRagQuery(message, message.author.id, message.content, 'canal', '[Consulta]');
      return;
    }

    // 3. DMs (Privadas) - Python RAG
    if (message.channel.type === ChannelType.DM) {
      console.log(`[DM] ${message.author.username} (${message.author.id}): ${message.content.substring(0, 50)}...`);
      await handleRagQuery(message, message.author.id, message.content, 'dm', '[DM]');
      return;
    }

  } catch (error) {
    console.error('[Error]', error);
  }
});

// Event: Reação adicionada (feedback 👍/👎)
client.on('messageReactionAdd', async (reaction, user) => {
  try {
    if (user.bot) return;

    // Reações em DM/mensagens não cacheadas vêm como partial
    if (reaction.partial) {
      try { await reaction.fetch(); } catch (e) { return; }
    }

    const interactionId = feedbackMap.get(reaction.message.id);
    if (!interactionId) return;

    let feedback = null;
    if (reaction.emoji.name === '✅') feedback = 'positivo';
    else if (reaction.emoji.name === '❌') feedback = 'negativo';
    if (!feedback) return;

    await axios.post(FEEDBACK_API_URL, { interaction_id: interactionId, feedback }, { timeout: 10000 });
    console.log(`[Feedback] ${feedback} registrado para ${interactionId}`);

  } catch (error) {
    console.error(`[Feedback] ❌ Erro: ${error.message}`);
  }
});

// Erro do cliente
client.on('error', error => {
  console.error('[Client Error]', error);
});

// Unhandled rejection
process.on('unhandledRejection', error => {
  console.error('[Unhandled Rejection]', error);
});

// Login
const token = process.env.DISCORD_BOT_TOKEN;
client.login(token);

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n[Bot] Desligando...');
  client.destroy();
  process.exit(0);
});
