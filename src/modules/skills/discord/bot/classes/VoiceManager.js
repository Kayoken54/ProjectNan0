const {
    joinVoiceChannel,
    getVoiceConnection,
    VoiceConnectionStatus,
    EndBehaviorType,
    createAudioPlayer,
    entersState,
    createAudioResource,
    StreamType,
    AudioPlayerStatus
} = require('@discordjs/voice');
const prism = require('prism-media');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const { Readable } = require('stream');

class VoiceManager {
    constructor(client) {
        this.client = client;
        this.connections = new Map(); // guildId -> connection data
        // [Discord Bridge] Legacy Brain HTTP is optional only.
        // Default runtime writes real Discord audio captures to disk instead of
        // calling a dead :8000 backend or inventing fake transcripts.
        this.apiBaseUrl = process.env.BRAIN_API_URL || '';
        this.audioInboxPath = process.env.DISCORD_AUDIO_INBOX_PATH || process.env.DISCORD_INBOX_PATH || path.resolve(process.cwd(), '../../../../data/input/discord_voice_inbox.jsonl');
        this.audioDir = process.env.DISCORD_AUDIO_DIR || path.resolve(process.cwd(), '../../../../data/input/discord_audio');

        // sustained-speech threshold: only interrupt Nan0 if someone talks for this long
        // each frame is ~20ms, so 2000ms = 100 frames
        this.INTERRUPT_THRESHOLD_MS = 2000;

        this.voiceOutboxPath = process.env.DISCORD_VOICE_OUTBOX_PATH || path.resolve(process.cwd(), '../../../../data/output/discord_voice_outbox.jsonl');
        this.voiceOutboxPosition = 0;
        this.voiceOutboxTimer = null;
        this.startVoiceOutboxWatcher();
    }

    isVoiceChannel(channel) {
        if (!channel) return false;
        if (typeof channel.isVoiceBased === 'function' && channel.isVoiceBased()) return true;
        // Discord channel type: 2 = GuildVoice, 13 = GuildStageVoice
        return channel.type === 2 || channel.type === 13;
    }

    channelHasHumans(channel) {
        try {
            if (!channel || !channel.members) return false;
            for (const member of channel.members.values()) {
                if (member && member.user && !member.user.bot) return true;
            }
        } catch (e) {}
        return false;
    }

    async resolveVoiceChannel(guild, requestedChannelId) {
        const envVoiceId = process.env.DISCORD_VOICE_CHANNEL_ID || process.env.TARGET_VOICE_CHANNEL_ID || '';
        const envVoiceName = process.env.DISCORD_VOICE_CHANNEL_NAME || process.env.DISCORD_TARGET_CHANNEL || process.env.TARGET_CHANNEL || '';

        const tryFetch = async (id, label) => {
            if (!id) return null;
            try {
                const ch = await guild.channels.fetch(id);
                if (this.isVoiceChannel(ch)) {
                    console.log(`[VoiceManager] Voice channel resolved by ${label}: ${ch.name || ch.id} (${ch.id})`);
                    return ch;
                }
                if (ch) console.warn(`[VoiceManager] ${label} channel is not voice: ${ch.name || ch.id} type=${ch.type}`);
            } catch (e) {
                console.warn(`[VoiceManager] Could not fetch ${label} channel ${id}: ${e.message}`);
            }
            return null;
        };

        let channel = await tryFetch(requestedChannelId, 'requested');
        if (channel) return channel;

        channel = await tryFetch(envVoiceId, 'env-id');
        if (channel) return channel;

        try {
            await guild.channels.fetch();
        } catch (e) {
            console.warn(`[VoiceManager] Could not refresh guild channels: ${e.message}`);
        }

        const voiceChannels = Array.from(guild.channels.cache.values()).filter(ch => this.isVoiceChannel(ch));

        if (envVoiceName) {
            const byName = voiceChannels.find(ch => String(ch.name || '').toLowerCase() === String(envVoiceName).toLowerCase());
            if (byName) {
                console.log(`[VoiceManager] Voice channel resolved by env-name: ${byName.name} (${byName.id})`);
                return byName;
            }
        }

        const occupied = voiceChannels.find(ch => this.channelHasHumans(ch));
        if (occupied) {
            console.log(`[VoiceManager] Voice channel resolved by occupied voice channel: ${occupied.name} (${occupied.id})`);
            return occupied;
        }

        const joinable = voiceChannels.find(ch => ch.joinable !== false);
        if (joinable) {
            console.log(`[VoiceManager] Voice channel resolved by first joinable voice channel: ${joinable.name} (${joinable.id})`);
            return joinable;
        }

        console.error(`[VoiceManager] No usable voice channel found. requestedChannelId=${requestedChannelId}`);
        return null;
    }

    async handleJoin(guildId, channelId, adapterCreator) {
        try {
            this.handleLeave(guildId, { silent: true });

            let guild;
            try {
                guild = await this.client.guilds.fetch(guildId);
                console.log(`[VoiceManager] Guild resolved: ${guild.name}`);
            } catch (e) {
                console.error(`[VoiceManager] Cannot fetch guild ${guildId}: ${e.message}`);
                return false;
            }

            const voiceChannel = await this.resolveVoiceChannel(guild, channelId);
            if (!voiceChannel) return false;

            const actualAdapterCreator = guild.voiceAdapterCreator || adapterCreator;
            if (!actualAdapterCreator) {
                console.error(`[VoiceManager] guild.voiceAdapterCreator is undefined!`);
                return false;
            }

            console.log(`[VoiceManager] Joining voice guild=${guildId} channel=${voiceChannel.id} name=${voiceChannel.name || 'unknown'}`);

            const connection = joinVoiceChannel({
                channelId: voiceChannel.id,
                guildId,
                adapterCreator: actualAdapterCreator,
                selfDeaf: false,
                selfMute: false,
            });

            const player = createAudioPlayer();
            connection.subscribe(player);

            const data = {
                connection,
                player,
                isSpeaking: false,
                subscriptions: new Map(),
                receiverArmed: false,
                receiverHandler: null,
                ready: false,
                guild,
                channelId: voiceChannel.id,
                requestedChannelId: channelId,
            };
            this.connections.set(guildId, data);

            player.on(AudioPlayerStatus.Playing, () => { data.isSpeaking = true; console.log('[VoiceManager] Nan0: SPEAKING'); });
            player.on(AudioPlayerStatus.Idle, () => { data.isSpeaking = false; console.log('[VoiceManager] Nan0: IDLE'); });
            player.on(AudioPlayerStatus.Paused, () => { data.isSpeaking = false; console.log('[VoiceManager] Nan0: PAUSED'); });
            player.on('error', (error) => { data.isSpeaking = false; console.error('[VoiceManager] Audio player error:', error.message); });

            connection.on('stateChange', (oldState, newState) => {
                console.log(`[VoiceManager] Voice state ${oldState.status} -> ${newState.status} guild=${guildId}`);
                if (newState.status === VoiceConnectionStatus.Ready) {
                    data.ready = true;
                    this.armVoiceReceiver(guildId, 'stateChange-ready');
                }
                if (newState.status === VoiceConnectionStatus.Disconnected) {
                    console.warn(`[VoiceManager] Disconnected guild=${guildId}`);
                    setTimeout(() => {
                        const existing = this.connections.get(guildId);
                        if (existing && existing.connection === connection && existing.connection.state.status !== VoiceConnectionStatus.Ready) {
                            console.log(`[VoiceManager] Rejoin attempt after disconnect guild=${guildId}`);
                            this.handleJoin(guildId, voiceChannel.id, actualAdapterCreator).catch(() => {});
                        }
                    }, 5000);
                }
                if (newState.status === VoiceConnectionStatus.Destroyed) {
                    this.cleanup(guildId);
                }
            });

            connection.on('error', (error) => {
                console.error(`[VoiceManager] Voice connection error guild=${guildId}:`, error.message);
            });

            try {
                await entersState(connection, VoiceConnectionStatus.Ready, 45000);
                data.ready = true;
                this.armVoiceReceiver(guildId, 'entersState-ready');
                console.log(`[VoiceManager] Voice connection READY guild=${guildId} channel=${voiceChannel.id}`);
                return true;
            } catch (error) {
                const status = connection.state?.status || 'unknown';
                console.error(`[VoiceManager] Timeout after 45s, status=${status}: ${error.message}`);
                console.error(`[VoiceManager] Requested channel was ${channelId}; resolved voice channel was ${voiceChannel.id}. If this still hangs, Discord voice networking or @discordjs/voice dependencies are the next suspects.`);
                this.handleLeave(guildId, { silent: true });
                return false;
            }
        } catch (error) {
            console.error('[VoiceManager] Error joining:', error);
            return false;
        }
    }

    handleLeave(guildId, opts = {}) {
        const data = this.connections.get(guildId);
        if (data && data.connection) {
            data.connection.destroy();
        }
        this.cleanup(guildId);
    }

    cleanup(guildId) {
        const data = this.connections.get(guildId);
        if (data) {
            if (data.player) data.player.stop();
            if (data.connection && data.connection.receiver && data.receiverHandler) {
                try { data.connection.receiver.speaking.off('start', data.receiverHandler); } catch (e) {}
            }
            // stop all streams
            for (const [userId, stream] of data.subscriptions) {
                stream.destroy();
            }
            this.connections.delete(guildId);
        }
    }

    listenToUsers(guildId) {
        return this.armVoiceReceiver(guildId, 'listenToUsers');
    }

    armVoiceReceiver(guildId, reason = 'unknown') {
        const data = this.connections.get(guildId);
        if (!data || !data.connection) return false;
        if (data.receiverArmed) {
            console.log(`[VoiceManager] Receiver already armed guild=${guildId}`);
            return true;
        }
        if (data.connection.state.status !== VoiceConnectionStatus.Ready) {
            console.warn(`[VoiceManager] Connection not Ready, deferring receiver arm guild=${guildId} status=${data.connection.state.status}`);
            const checkReady = setInterval(() => {
                const current = this.connections.get(guildId);
                if (!current || !current.connection) {
                    clearInterval(checkReady);
                    return;
                }
                if (current.connection.state.status === VoiceConnectionStatus.Ready) {
                    clearInterval(checkReady);
                    this.armVoiceReceiver(guildId, 'deferred-ready');
                }
            }, 1000);
            setTimeout(() => clearInterval(checkReady), 30000);
            return false;
        }

        const receiver = data.connection.receiver;
        if (!receiver || !receiver.speaking) {
            console.error(`[VoiceManager] No receiver/speaking object guild=${guildId}`);
            return false;
        }

        data.receiverHandler = (userId) => {
            if (data.subscriptions.has(userId)) return;
            if (this.client.user && userId === this.client.user.id) return;
            console.log(`[VoiceManager] User ${userId} started speaking`);
            this.createStream(guildId, userId);
        };
        receiver.speaking.on('start', data.receiverHandler);
        data.receiverArmed = true;
        console.log(`[VoiceManager] Voice receiver armed guild=${guildId} reason=${reason}`);
        return true;
    }

    createStream(guildId, userId) {
        const data = this.connections.get(guildId);
        if (!data) return;

        const opusStream = data.connection.receiver.subscribe(userId, {
            end: {
                behavior: EndBehaviorType.AfterSilence,
                duration: 100, // fast but safe: opus frames are 20ms, need margin to not clip words
            },
        });

        // save stream
        data.subscriptions.set(userId, opusStream);

        // decode opus to pcm (signed 16-bit little endian, 48khz, stereo)
        const decoder = new prism.opus.Decoder({ frameSize: 960, channels: 2, rate: 48000 });
        const pcmStream = opusStream.pipe(decoder);

        const chunks = [];

        // vad state: noise gate
        let speechFrameCount = 0;
        const VAD_THRESHOLD = 800; // ignore typing clicks / background noise
        const MIN_SPEECH_FRAMES = 6; // require 120ms of sustained volume (6 * 20ms)

        // sustained-speech interrupt detection
        // each frame is ~20ms. we track if the user has been speaking long enough to interrupt Nan0.
        const interruptFrameThreshold = Math.floor(this.INTERRUPT_THRESHOLD_MS / 20);
        let didInterrupt = false;

        const Nan0WasSpeaking = data.isSpeaking;

        pcmStream.on('data', (chunk) => {
            chunks.push(chunk);

            // analyze energy
            const rms = this.calculateRMS(chunk);
            if (rms > VAD_THRESHOLD) {
                speechFrameCount++;

                // live interrupt check: only if Nan0 is currently playing audio
                if (!didInterrupt && data.isSpeaking && speechFrameCount >= interruptFrameThreshold) {
                    console.log(`[VoiceManager] Sustained speech (${(speechFrameCount * 20 / 1000).toFixed(1)}s) — INTERRUPTING Nan0`);
                    data.player.stop();
                    if (this.hasLegacyBrainApi()) axios.post(`${this.apiBaseUrl}/interrupt`).catch(e => { });
                    didInterrupt = true;
                }
            }
        });

        pcmStream.on('end', async () => {
            // clean up
            data.subscriptions.delete(userId);
            const speechDurationMs = speechFrameCount * 20;
            console.log(`[VoiceManager] Stream ended. Speech: ${speechDurationMs}ms (${speechFrameCount} frames), Nan0WasSpeaking=${Nan0WasSpeaking}, isSpeaking=${data.isSpeaking}`);

            // 1. noise filter: if audio was too short or too quiet
            if (speechFrameCount < MIN_SPEECH_FRAMES) {
                console.log("[VoiceManager] Discarding noise (Keyboard/Background).");
                return;
            }

            // 2. valid speech — determine how to handle it
            if (chunks.length === 0) return;

            const totalBuffer = Buffer.concat(chunks);

            // key logic: behavior depends on whether Nan0 was speaking
            if (!Nan0WasSpeaking && !data.isSpeaking) {
                // Nan0 is idle → process all valid speech immediately, no threshold needed
                console.log(`[VoiceManager] Nan0 is idle → sending to full LLM pipeline`);
                await this.processAudio(guildId, userId, totalBuffer, true);
            } else if (speechDurationMs >= this.INTERRUPT_THRESHOLD_MS || didInterrupt) {
                // Nan0 was speaking but user talked long enough to interrupt
                console.log(`[VoiceManager] Sustained speech interrupted Nan0 → full LLM pipeline`);
                await this.processAudio(guildId, userId, totalBuffer, true);
            } else {
                // Nan0 is speaking and user speech was short → buffer only
                console.log(`[VoiceManager] Short speech while Nan0 talks → buffering transcript`);
                await this.bufferTranscript(guildId, userId, totalBuffer);
            }
        });

        pcmStream.on('error', (err) => {
            console.error(`[VoiceManager] Stream error for ${userId}:`, err);
            data.subscriptions.delete(userId);
        });
    }

    calculateRMS(buffer) {
        let sum = 0;
        const len = buffer.length / 2;
        if (len === 0) return 0;

        for (let i = 0; i < buffer.length; i += 2) {
            const int16 = buffer.readInt16LE(i);
            sum += int16 * int16;
        }
        return Math.sqrt(sum / len);
    }


    // [Discord Bridge] Save real captured Discord voice audio for the Python runtime/STT layer.
    // This does not fabricate transcripts. It preserves real input and avoids the dead :8000 Brain API.
    saveAudioInboxRecord(guildId, userId, username, wavBuffer, mode = 'full') {
        try {
            fs.mkdirSync(this.audioDir, { recursive: true });
            fs.mkdirSync(path.dirname(this.audioInboxPath), { recursive: true });
            const now = Date.now();
            const safeUser = String(username || userId || 'unknown').replace(/[^a-z0-9_-]+/gi, '_').slice(0, 40) || 'unknown';
            const filename = `discord_${now}_${safeUser}.wav`;
            const audioPath = path.join(this.audioDir, filename);
            fs.writeFileSync(audioPath, wavBuffer);
            const record = {
                source: 'discord_voice',
                speaker: username || userId || 'unknown',
                source_actor_id: userId || username || 'discord_voice',
                user_id: userId,
                guild_id: guildId,
                audio_path: audioPath,
                text: '',
                addressed_to_nan0: false,
                needs_transcription: true,
                mode,
                timestamp: now / 1000
            };
            fs.appendFileSync(this.audioInboxPath, JSON.stringify(record) + '\n', 'utf8');
            console.log(`[VoiceManager] Saved Discord voice audio for ${record.speaker}: ${audioPath}`);
            return record;
        } catch (error) {
            console.error('[VoiceManager] Failed to save Discord voice audio:', error.message);
            return null;
        }
    }

    hasLegacyBrainApi() {
        return !!(this.apiBaseUrl && /^https?:\/\//i.test(this.apiBaseUrl));
    }

    /**
     * buffer a short transcript without triggering llm.
     * transcribes locally then sends to /voice/transcript for accumulation.
     */
    async bufferTranscript(guildId, userId, pcmBuffer) {
        // 1. fetch username
        let username = userId;
        try {
            const guild = await this.client.guilds.fetch(guildId);
            const member = await guild.members.fetch(userId);
            username = member.displayName;
        } catch (e) {
            console.error("Error fetching user:", e);
        }

        // 2. convert pcm to wav
        const wavBuffer = this.pcmToWav(pcmBuffer, 48000, 2);

        // 3. Current Nan0 runtime has no :8000 voice transcript API by default.
        // Save real audio instead; only call legacy API if explicitly configured.
        this.saveAudioInboxRecord(guildId, userId, username, wavBuffer, 'buffer');
        if (!this.hasLegacyBrainApi()) {
            return;
        }

        // 4. Optional legacy endpoint.
        try {
            const form = new FormData();
            form.append('file', wavBuffer, { filename: 'audio.wav', contentType: 'audio/wav' });
            form.append('username', username);

            const response = await axios.post(`${this.apiBaseUrl}/voice/transcript`, form, {
                headers: { ...form.getHeaders() }
            });

            console.log(`[VoiceManager] Transcript buffered for ${username}: ${response.data.transcript || '(empty)'}`);
        } catch (error) {
            console.error("[VoiceManager] Buffer transcript error:", error.message);
        }
    }

    async processAudio(guildId, userId, pcmBuffer, flushBuffer = false) {
        const data = this.connections.get(guildId);

        // 1. fetch user info
        let username = userId;
        try {
            const guild = await this.client.guilds.fetch(guildId);
            const member = await guild.members.fetch(userId);
            username = member.displayName;
        } catch (e) {
            console.error("Error fetching user:", e);
        }

        console.log(`[VoiceManager] Processing audio from ${username} (${pcmBuffer.length} bytes, flush=${flushBuffer})`);

        // 2. convert pcm to wav
        const wavBuffer = this.pcmToWav(pcmBuffer, 48000, 2);

        // 3. Current Nan0 runtime has no :8000 /discord/audio API by default.
        // Save real audio for a future STT consumer; do not invent transcript text.
        this.saveAudioInboxRecord(guildId, userId, username, wavBuffer, flushBuffer ? 'full_flush' : 'full');
        if (!this.hasLegacyBrainApi()) {
            console.log('[VoiceManager] No BRAIN_API_URL configured; audio capture saved, no legacy backend call.');
            return;
        }

        // 4. Optional legacy endpoint.
        try {
            const form = new FormData();
            form.append('file', wavBuffer, { filename: 'audio.wav', contentType: 'audio/wav' });
            form.append('username', username);
            if (flushBuffer) {
                form.append('flush_buffer', 'true');
            }

            const response = await axios.post(`${this.apiBaseUrl}/discord/audio`, form, {
                headers: { ...form.getHeaders() }
            });

            const { status, text, audio_base64, thought_id } = response.data;
            const hasThoughtId = typeof thought_id === 'string' && thought_id.trim().length > 0;

            if (status === 'success') {
                // new response -> stop old, play new only when the Python side proves thought origin
                if (data && data.player) {
                    data.player.stop(); // Stop potential paused content
                    console.log(`[VoiceManager] Response: "${text}" thought_id=${thought_id || 'missing'}`);
                    if (audio_base64 && hasThoughtId) {
                        this.playAudio(guildId, audio_base64);
                    } else if (audio_base64 && !hasThoughtId) {
                        console.error('[VoiceManager] Blocked legacy audio playback: missing thought_id.');
                    }
                }
            } else if (status === 'resume') {
                // backchannel -> resume old content
                console.log(`[VoiceManager] Backchannel detected: "${text}". Resuming...`);
                if (data && data.player && data.player.state.status === AudioPlayerStatus.Paused) {
                    data.player.unpause();
                }
            }

        } catch (error) {
            console.error("[VoiceManager] API Error:", error.message);
            if (data && data.player && data.player.state.status === AudioPlayerStatus.Paused) {
                data.player.unpause();
            }
        }
    }

    playAudio(guildId, base64Audio) {
        const data = this.connections.get(guildId);
        if (!data || !data.player) return;

        try {
            const buffer = Buffer.from(base64Audio, 'base64');

            // create readable stream
            const stream = Readable.from(buffer);

            const resource = createAudioResource(stream, {
                inputType: StreamType.Arbitrary
            });

            data.player.play(resource);

        } catch (e) {
            console.error("[VoiceManager] Playback Logic Error:", e);
        }
    }


    startVoiceOutboxWatcher() {
        try {
            fs.mkdirSync(path.dirname(this.voiceOutboxPath), { recursive: true });
            if (!fs.existsSync(this.voiceOutboxPath)) fs.writeFileSync(this.voiceOutboxPath, '', 'utf8');
            this.voiceOutboxPosition = fs.statSync(this.voiceOutboxPath).size;
            console.log(`[VoiceManager] Watching voice outbox: ${this.voiceOutboxPath}`);
        } catch (e) {
            console.error('[VoiceManager] Failed to initialize voice outbox watcher:', e.message);
        }

        if (this.voiceOutboxTimer) clearInterval(this.voiceOutboxTimer);
        this.voiceOutboxTimer = setInterval(() => this.pollVoiceOutbox(), 250);
    }

    pollVoiceOutbox() {
        try {
            if (!fs.existsSync(this.voiceOutboxPath)) return;
            const stat = fs.statSync(this.voiceOutboxPath);
            if (stat.size < this.voiceOutboxPosition) this.voiceOutboxPosition = 0;
            if (stat.size === this.voiceOutboxPosition) return;

            const fd = fs.openSync(this.voiceOutboxPath, 'r');
            const length = stat.size - this.voiceOutboxPosition;
            const buffer = Buffer.alloc(length);
            fs.readSync(fd, buffer, 0, length, this.voiceOutboxPosition);
            fs.closeSync(fd);
            this.voiceOutboxPosition = stat.size;

            const lines = buffer.toString('utf8').split(/\r?\n/).filter(Boolean);
            for (const line of lines) {
                try {
                    const record = JSON.parse(line);
                    if (record.type === 'play_audio_file' && record.audio_path) {
                        if (!record.thought_id) {
                            console.error('[VoiceManager] Blocked outbox audio playback: missing thought_id.');
                            continue;
                        }
                        this.playAudioFile(record.audio_path, record);
                    }
                } catch (e) {
                    console.error('[VoiceManager] Bad voice outbox record:', e.message);
                }
            }
        } catch (e) {
            console.error('[VoiceManager] Voice outbox poll failed:', e.message);
        }
    }

    playAudioFile(audioPath, record = {}) {
        const guildId = record.guild_id || this.firstReadyGuildId();
        const data = guildId ? this.connections.get(guildId) : null;
        if (!data || !data.player || data.connection.state.status !== VoiceConnectionStatus.Ready) {
            console.log(`[VoiceManager] No active Ready VC connection. Queued audio cannot play yet: ${audioPath}`);
            return false;
        }
        if (!fs.existsSync(audioPath)) {
            console.error(`[VoiceManager] Audio file missing: ${audioPath}`);
            return false;
        }
        try {
            const resource = createAudioResource(audioPath);
            console.log(`[VoiceManager] Playing VC audio guild=${guildId} thought_id=${record.thought_id || 'unknown'} file=${audioPath}`);
            data.player.play(resource);
            return true;
        } catch (e) {
            console.error('[VoiceManager] playAudioFile failed:', e.message);
            return false;
        }
    }

    firstReadyGuildId() {
        for (const [guildId, data] of this.connections.entries()) {
            if (data && data.connection && data.connection.state.status === VoiceConnectionStatus.Ready) return guildId;
        }
        return null;
    }

    // helper: add wav header
    pcmToWav(pcmData, sampleRate, numChannels) {
        const header = Buffer.alloc(44);
        const byteRate = sampleRate * numChannels * 2; // 16-bit = 2 bytes
        const blockAlign = numChannels * 2;
        const subChunk2Size = pcmData.length;
        const chunkSize = 36 + subChunk2Size;

        header.write('RIFF', 0);
        header.writeUInt32LE(chunkSize, 4);
        header.write('WAVE', 8);

        header.write('fmt ', 12);
        header.writeUInt32LE(16, 16);
        header.writeUInt16LE(1, 20);
        header.writeUInt16LE(numChannels, 22);
        header.writeUInt32LE(sampleRate, 24);
        header.writeUInt32LE(byteRate, 28);
        header.writeUInt16LE(blockAlign, 32);
        header.writeUInt16LE(16, 34);

        header.write('data', 36);
        header.writeUInt32LE(subChunk2Size, 40);

        return Buffer.concat([header, pcmData]);
    }
}

module.exports = VoiceManager;
