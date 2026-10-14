// أزوم الزكي — internal team assistant.
// POST /api/chat  { password, check?: true }            -> { ok: true } when the password is right
// POST /api/chat  { password, messages: [{role, content}] } -> streamed NDJSON events:
//   {"t":"status","v":"searching"|"checking"} | {"t":"text","v":"..."} | {"t":"reset"} | {"t":"error","v":"..."} | {"t":"done"}
//   "reset" means: discard text shown so far (the model wrote something before looking up data).
// Needs ANTHROPIC_API_KEY and TEAM_PASSWORD in the environment (Vercel settings, or .env.local locally).
import crypto from 'node:crypto';
import Anthropic from '@anthropic-ai/sdk';
import { KNOWLEDGE } from './_knowledge.js';
import { PROGRAMS } from './_programs.js';

const MODEL = 'claude-haiku-5-5'; // cheapest; matched Opus/Sonnet on fee look-ups in our test (2026-10-14)
const MAX_MESSAGES = 20;       // conversation turns kept per request
const MAX_CHARS = 4000;        // per message
const MAX_STEPS = 8;           // model calls per answer (tool look-ups and web-search pauses)
const WRONG_LIMIT = 5;         // wrong passwords per IP ...
const WRONG_WINDOW_MS = 15 * 60 * 1000; // ... per 15 minutes

export const SYSTEM = `You are "أزوم الزكي" (Smart Azoom), the internal assistant of the Azoom team. Azoom is a study-abroad agency that helps students study in Malaysia, Egypt, Rwanda and other destinations (China, Turkey, Indonesia). The people talking to you are Azoom staff, not students — help them answer student enquiries quickly and accurately.

Sources:
- Programme names and fees: ALWAYS use the search_programs tool, for every question that involves a programme, a fee, a ranking ("cheapest", "most expensive") or a comparison. It searches Azoom's complete, exact website data and does the sorting for you. Use only the figures it returns — never from memory and never by estimating. Call it more than once when the question needs it (for example one call per university or per field). Do not write anything before your tool calls.
- Each programme has ONE fee: fee_usd, the amount the student actually pays (after any scholarship or discount). Show only that number, in one fee column. Never show a full/original price, never add a second price column, and do not mention scholarships or discounts unless the user asks about them.
- Lists across universities show each university once (the tool does this by default), even when the question says "programmes".
- The knowledge below is Azoom's website text: universities, services, processes, Egypt and Rwanda details. Fees are in US dollars per year unless stated otherwise.
- For general questions the data does not cover (visas, cost of living, scholarships, rankings), search the web.
- Never invent fees, programmes, deadlines or requirements. If something is not available, say so in one short sentence. Never promise admission, scholarships or visa approval.

Writing style — the team copies your answers straight to students, so every answer must be ready to send:
- Always write in clear, natural Modern Standard Arabic, whatever language the question is in. Copy the university and programme names the tool returns exactly as written, letter for letter — do not rephrase, re-translate or add the English name. If a returned name is in English, translate it into Arabic; never leave English words inside an Arabic name. Never add an abbreviation in brackets after a name (write "جامعة بيرليس", not "جامعة بيرليس (UniMAP)"), and drop "ماليزيا" from Malaysian university names.
- Start directly with the answer. No greeting, no introduction, no restating the question, no closing offer, no emojis.
- Questions about programmes, fees or comparisons between universities or countries: answer with one compact Markdown table — at most 4 short columns, Arabic headers, amounts written like "6,434 دولار". Order rows in the most useful way (cheapest first when cost matters). Add at most one short line after the table, and only if it is essential.
- General questions: one or two short, well-written paragraphs.
- Steps or required documents: a short numbered list.
- Keep it concise. Avoid brackets, footnotes and side remarks unless they are truly necessary. Use bold only for the key figure or name, never italics.

=== AZOOM WEBSITE KNOWLEDGE ===
${KNOWLEDGE}`;

const client = new Anthropic(); // reads ANTHROPIC_API_KEY
const wrongByIp = new Map();

// ---------- search_programs: exact look-ups over the website's programme data ----------
const CATEGORIES = [...new Set(PROGRAMS.map(p => p.category))].sort();
const COUNTRIES = [...new Set(PROGRAMS.map(p => p.country))];

const SEARCH_PROGRAMS_TOOL = {
  name: 'search_programs',
  description: `Searches Azoom's complete website data of ${PROGRAMS.length} programmes with exact fees (USD) and returns matching rows, filtered and sorted exactly. The data is in English, so use English for keyword and university. "fee_usd" is the one fee to quote: what the student pays, already after any scholarship or discount. Fees are per year unless fee_basis says otherwise. Without "university", each university appears once (its best match); with "university", all its matching programmes are returned.`,
  input_schema: {
    type: 'object',
    properties: {
      country: { type: 'string', enum: [...COUNTRIES, 'any'], description: 'Country to search. Default: any.' },
      category: { type: 'string', enum: CATEGORIES, description: 'Field of study.' },
      keyword: { type: 'string', description: 'Words that must all appear in the programme name. English works best (e.g. "civil engineering", "pharmacy", "MBA").' },
      university: { type: 'string', description: 'Part of the university name or abbreviation, in English (e.g. "UCSI", "Taylor", "Lincoln", "Perlis") or as written in Arabic in earlier results.' },
      max_fee: { type: 'number', description: 'Only programmes whose effective fee is at most this (USD).' },
      min_fee: { type: 'number', description: 'Only programmes whose effective fee is at least this (USD).' },
      sort: { type: 'string', enum: ['cheapest', 'most_expensive', 'none'], description: 'Default: cheapest.' },
      limit: { type: 'integer', minimum: 1, maximum: 60, description: 'Maximum rows to return. Default: 15.' },
    },
    additionalProperties: false,
  },
  eager_input_streaming: true,
};

const norm = s => String(s || '').toLowerCase().replace(/[’']/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
const effective = p => p.fee_after_discount_usd ?? p.annual_fee_usd;

export function searchPrograms(q) {
  const isArabic = s => /[؀-ۿ]/.test(s || '');
  const keyword = String(q.keyword || '').trim(), uniQuery = String(q.university || '').trim();
  const words = isArabic(keyword) ? keyword.split(/\s+/).filter(Boolean) : norm(keyword).split(' ').filter(Boolean);
  const uni = uniQuery; // non-empty means "search inside one university"
  const uniMatches = p => isArabic(uniQuery) ? (p.university_ar || '').includes(uniQuery) : norm(p.university).includes(norm(uniQuery));
  const wordMatches = p => isArabic(keyword) ? words.every(w => (p.programme_ar || '').includes(w)) : words.every(w => norm(p.programme).includes(w));
  let rows = PROGRAMS.filter(p =>
    (!q.country || q.country === 'any' || p.country === q.country) &&
    (!q.category || p.category === q.category) &&
    (!uni || uniMatches(p)) &&
    wordMatches(p) &&
    (q.max_fee == null || effective(p) <= q.max_fee) &&
    (q.min_fee == null || effective(p) >= q.min_fee));
  const sort = q.sort || 'cheapest';
  if (sort === 'cheapest') rows.sort((a, b) => effective(a) - effective(b) || a.annual_fee_usd - b.annual_fee_usd);
  if (sort === 'most_expensive') rows.sort((a, b) => effective(b) - effective(a));
  // Lists across universities show each university once (team's preference); searching inside one university shows all its programmes.
  if (!uni) { const seen = new Set(); rows = rows.filter(p => !seen.has(p.university) && seen.add(p.university)); }
  const total = rows.length;
  const limit = Math.min(Math.max(Number(q.limit) || 15, 1), 60);
  return {
    total_matches: total,
    returned: Math.min(total, limit),
    // One fee per programme (after scholarship/discount); full prices are left out on purpose.
    // Names are the site's official Arabic names when available, so the model has nothing to translate or mix.
    results: rows.slice(0, limit).map(p => ({
      country: p.country, university: p.university_ar || p.university, programme: p.programme_ar || p.programme,
      university_type: p.university_type, city: p.city, category: p.category, fee_basis: p.fee_basis, fee_usd: effective(p),
    })),
    ...(total ? {} : { hint: `No match. Valid categories: ${CATEGORIES.join(', ')}. Try fewer keywords or a different spelling.` }),
  };
}

function runTool(block) {
  const input = block.input;
  if (block.name !== 'search_programs') return { content: `Unknown tool ${block.name}`, is_error: true };
  if (!input || typeof input !== 'object' || Array.isArray(input)) return { content: 'Invalid input: expected a JSON object.', is_error: true };
  const known = Object.fromEntries(Object.entries(input).filter(([k]) => k in SEARCH_PROGRAMS_TOOL.input_schema.properties));
  return { content: JSON.stringify(searchPrograms(known)) };
}

function clientIp(req) {
  return String(req.headers['x-forwarded-for'] || req.socket?.remoteAddress || '').split(',')[0].trim();
}

function tooManyWrong(ip) {
  const now = Date.now();
  const hits = (wrongByIp.get(ip) || []).filter(t => now - t < WRONG_WINDOW_MS);
  wrongByIp.set(ip, hits);
  return hits.length >= WRONG_LIMIT;
}

// Copying through WhatsApp/phones can add spaces or invisible direction marks; ignore them on both sides.
const cleanSecret = s => String(s ?? '').replace(/[​-‏‪-‮⁦-⁩﻿]/g, '').trim();

function passwordOk(given) {
  const expected = cleanSecret(process.env.TEAM_PASSWORD);
  if (!expected || typeof given !== 'string') return false;
  const hash = s => crypto.createHash('sha256').update(s, 'utf8').digest();
  return crypto.timingSafeEqual(hash(cleanSecret(given)), hash(expected));
}

function cleanMessages(raw) {
  if (!Array.isArray(raw)) return null;
  const msgs = raw
    .filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim())
    .map(m => ({ role: m.role, content: m.content.slice(0, MAX_CHARS) }))
    .slice(-MAX_MESSAGES);
  while (msgs.length && msgs[0].role !== 'user') msgs.shift();
  if (!msgs.length || msgs[msgs.length - 1].role !== 'user') return null;
  return msgs;
}

function json(res, status, body) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(body));
}

function errorText(err) {
  if (err instanceof Anthropic.AuthenticationError) return 'مفتاح الخدمة على الخادم غير صالح (ANTHROPIC_API_KEY).';
  if (err instanceof Anthropic.PermissionDeniedError) return 'حساب الخدمة لا يملك صلاحية أو رصيدًا كافيًا.';
  if (err instanceof Anthropic.RateLimitError) return 'طلبات كثيرة في الوقت نفسه. انتظر دقيقة ثم حاول مجددًا.';
  if (err instanceof Anthropic.APIError) return `حدث خطأ في خدمة الذكاء الاصطناعي (${err.status ?? 'network'}). حاول مجددًا.`;
  return 'حدث خطأ غير متوقع. حاول مجددًا.';
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return json(res, 405, { error: 'Method not allowed' });

  const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : (req.body || {});
  // Not a wrong password: the server itself is missing settings. Say so, and don't lock anyone out.
  if (!cleanSecret(process.env.TEAM_PASSWORD) || !process.env.ANTHROPIC_API_KEY) {
    return json(res, 503, { error: 'not_configured' });
  }
  const ip = clientIp(req);
  if (tooManyWrong(ip)) return json(res, 429, { error: 'Too many wrong passwords. Try again in 15 minutes.' });
  if (!passwordOk(body.password)) {
    wrongByIp.get(ip).push(Date.now());
    return json(res, 401, { error: 'Wrong password' });
  }
  if (body.check) return json(res, 200, { ok: true });

  let messages = cleanMessages(body.messages);
  if (!messages) return json(res, 400, { error: 'Send at least one message.' });

  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Accel-Buffering', 'no');
  const send = ev => res.write(JSON.stringify(ev) + '\n');

  const abort = new AbortController();
  res.on('close', () => abort.abort());

  try {
    for (let step = 0; step < MAX_STEPS; step++) {
      // Haiku has no server-side refusal fallback and uses the basic web search tool.
      const stream = client.messages.stream({
        model: MODEL,
        max_tokens: 16000,
        output_config: { effort: 'low' }, // look-ups and comparisons; keeps replies quick
        system: [{ type: 'text', text: SYSTEM, cache_control: { type: 'ephemeral', ttl: '1h' } }],
        tools: [SEARCH_PROGRAMS_TOOL, { type: 'web_search_20250305', name: 'web_search', max_uses: 3 }],
        messages,
      }, { signal: abort.signal });

      let wroteText = false;
      for await (const ev of stream) {
        if (ev.type === 'content_block_start' && ev.content_block.type === 'server_tool_use') send({ t: 'status', v: 'searching' });
        else if (ev.type === 'content_block_start' && ev.content_block.type === 'tool_use') send({ t: 'status', v: 'checking' });
        else if (ev.type === 'content_block_delta' && ev.delta.type === 'text_delta') { wroteText = true; send({ t: 'text', v: ev.delta.text }); }
      }
      const final = await stream.finalMessage();
      const toolUses = final.content.filter(b => b.type === 'tool_use');

      if (final.stop_reason === 'pause_turn') {
        messages = [...messages, { role: 'assistant', content: final.content }];
        continue;
      }
      if (final.stop_reason === 'tool_use' && toolUses.length) {
        if (wroteText) send({ t: 'reset' }); // text before a look-up is not part of the answer
        const results = toolUses.map(b => ({ type: 'tool_result', tool_use_id: b.id, ...runTool(b) }));
        messages = [...messages, { role: 'assistant', content: final.content }, { role: 'user', content: results }];
        continue;
      }
      if (final.stop_reason === 'refusal') send({ t: 'error', v: 'تعذّرت الإجابة عن هذا الطلب. جرّب صياغة السؤال بطريقة أخرى.' });
      else if (final.stop_reason === 'max_tokens') send({ t: 'error', v: 'انقطعت الإجابة لأنها طويلة جدًا. اطلب إجابة أقصر.' });
      break;
    }
    send({ t: 'done' });
  } catch (err) {
    if (!abort.signal.aborted) {
      console.error('chat error:', err?.status, err?.message);
      send({ t: 'error', v: errorText(err) });
    }
  }
  res.end();
}
