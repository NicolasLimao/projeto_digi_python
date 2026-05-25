const { Client, GatewayIntentBits, ChannelType, Partials } = require('discord.js');
const axios = require('axios');
require('dotenv').config();

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.DirectMessages,
    GatewayIntentBits.MessageContent
  ],
  partials: [Partials.Channel, Partials.Message]
});

// URLs e IDs
const N8N_WEBHOOK_INGESTAO = process.env.N8N_WEBHOOK_INGESTAO || 'http://localhost:5678/webhook/digi-ingestao';
const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000/api/rag/query';

const CANAL_INGESTAO = '1491637301522989198';
const CANAL_CONSULTA = '1491637352513142914';

console.log('[Bot] Iniciando...');
console.log('[Bot] RAG API URL:', RAG_API_URL);
console.log('[Bot] N8N Ingestão URL:', N8N_WEBHOOK_INGESTAO);

// Event: Bot conecta
client.on('ready', () => {
  console.log(`[Bot] ✅ Conectado como ${client.user.tag}`);
  console.log('[Bot] Escutando mensagens...');
});

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

      // Enviar para n8n (ingestão)
      try {
        await axios.post(N8N_WEBHOOK_INGESTAO, payload, { timeout: 5000 });
        console.log(`[Ingestao] ✅ Enviado para n8n`);
      } catch (error) {
        console.error(`[Ingestao] ❌ Erro: ${error.message}`);
      }
      return;
    }

    // 2. CANAL DE CONSULTA (Dúvidas) - Novo: Python RAG
    if (message.channelId === CANAL_CONSULTA) {
      console.log(`[Consulta] ${message.author.username}: ${message.content.substring(0, 50)}...`);

      await message.channel.sendTyping();

      // Chamar RAG API
      try {
        const url = new URL(RAG_API_URL);
        url.searchParams.append('user_id', message.author.id);

        const response = await axios.post(url.toString(), {
          query: message.content
        }, { timeout: 30000 });

        const result = response.data;
        const responseText = result.response || 'Sem resposta';
        const score = (result.score || 0).toFixed(2);
        const chunks = result.chunks_used || 0;
        const time = result.processing_time_ms || 0;

        console.log(`[Consulta] ✅ Resposta (score=${score}, chunks=${chunks}, time=${time}ms)`);

        // Enviar resposta
        if (responseText.length > 1900) {
          const msgChunks = [];
          let currentChunk = '';

          responseText.split('\n').forEach((line) => {
            if ((currentChunk + line).length > 1900) {
              msgChunks.push(currentChunk);
              currentChunk = line;
            } else {
              currentChunk += (currentChunk ? '\n' : '') + line;
            }
          });

          if (currentChunk) msgChunks.push(currentChunk);

          for (const chunk of msgChunks) {
            await message.reply(chunk);
          }

          console.log(`[Consulta] Enviadas ${msgChunks.length} mensagens`);
        } else {
          await message.reply(responseText);
          console.log(`[Consulta] Resposta enviada`);
        }

      } catch (error) {
        console.error(`[Consulta] ❌ Erro: ${error.message}`);

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
      return;
    }

    // 3. DMs (Privadas) - Novo: Python RAG
    if (message.channel.type === ChannelType.DM) {
      const userId = message.author.id;
      const userName = message.author.username;
      const userQuery = message.content;

      console.log(`[DM] ${userName} (${userId}): ${userQuery.substring(0, 50)}...`);

      await message.channel.sendTyping();

      try {
        const url = new URL(RAG_API_URL);
        url.searchParams.append('user_id', userId);

        const response = await axios.post(url.toString(), {
          query: userQuery
        }, { timeout: 30000 });

        const result = response.data;
        const responseText = result.response || 'Sem resposta';
        const score = (result.score || 0).toFixed(2);
        const chunks = result.chunks_used || 0;
        const time = result.processing_time_ms || 0;

        console.log(`[DM] ✅ Resposta (score=${score}, chunks=${chunks}, time=${time}ms)`);

        // Enviar resposta
        if (responseText.length > 1900) {
          const msgChunks = [];
          let currentChunk = '';

          responseText.split('\n').forEach((line) => {
            if ((currentChunk + line).length > 1900) {
              msgChunks.push(currentChunk);
              currentChunk = line;
            } else {
              currentChunk += (currentChunk ? '\n' : '') + line;
            }
          });

          if (currentChunk) msgChunks.push(currentChunk);

          for (const chunk of msgChunks) {
            await message.reply(chunk);
          }

          console.log(`[DM] Enviadas ${msgChunks.length} mensagens para ${userName}`);
        } else {
          await message.reply(responseText);
          console.log(`[DM] ✅ Resposta enviada para ${userName}`);
        }

      } catch (error) {
        console.error(`[DM] ❌ Erro: ${error.message}`);

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

  } catch (error) {
    console.error('[Error]', error);
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
const token = process.env.DISCORD_BOT_TOKEN || 'MTQ4NjcxMDg5OTYzNjA0Mzc4Ng.Go7IzE.4hESF1GDlAm27MjqAixPqqIJrsoNTC0iCvJqD0';
client.login(token);

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n[Bot] Desligando...');
  client.destroy();
  process.exit(0);
});
