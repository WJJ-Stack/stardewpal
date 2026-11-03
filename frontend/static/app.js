/* StardewPal 前端：无框架，原生 JS + SSE */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const messages = $('messages');
  const input = $('input');
  const sendBtn = $('send');
  const fileInput = $('fileInput');
  const preview = $('preview');
  const previewImg = $('previewImg');
  const previewName = $('previewName');
  const previewSize = $('previewSize');
  const removeImg = $('removeImg');

  let pendingFile = null;
  let busy = false;
  // 记住最后一次发送的图片：追问时自动带上，
  // 否则「那我刚给的图里是什么」这类问题后端拿不到图
  let lastImage = null;      // { blob, url }
  let lastImageUsedAt = 0;   // 上次使用的时间戳
  // 多轮对话历史：只保留最近 6 轮，避免请求体过大
  const history = [];
  const HISTORY_LIMIT = 6;

  function pushHistory(question, answer) {
    const q = (question || '').trim();
    const a = (answer || '').trim();
    if (!q || !a) return;
    history.push({ question: q.slice(0, 500), answer: a.slice(0, 2000) });
    while (history.length > HISTORY_LIMIT) history.shift();
  }

  // ------------------------------------------------------------ 状态
  async function loadStatus() {
    try {
      const r = await fetch('/health');
      const d = await r.json();
      const dot = $('statusDot');
      const txt = $('statusText');
      if (d.status === 'ok') {
        dot.className = 'dot ok';
        txt.textContent = d.llm_live ? '在线 · 模型已连接' : '在线 · 演示模式';
        if (!d.llm_live) dot.className = 'dot warn';
      } else {
        dot.className = 'dot bad';
        txt.textContent = '索引为空';
      }
      $('mDocs').textContent = d.index.docs ?? '—';
      $('mChunks').textContent = d.index.chunks ?? '—';
      $('mItems').textContent = d.index.items ?? '—';
    } catch {
      $('statusDot').className = 'dot bad';
      $('statusText').textContent = '服务不可用';
    }
  }

  async function loadMeta() {
    try {
      const r = await fetch('/api/meta');
      const d = await r.json();
      const icons = d.corpus.icons ?? 0;
      $('mIcons').textContent = icons || '—';
    } catch { /* 忽略 */ }
  }

  // ------------------------------------------------------------ 渲染
  function addUser(text, imageUrl) {
    const wrap = document.createElement('div');
    wrap.className = 'msg user';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (imageUrl) {
      const img = document.createElement('img');
      img.src = imageUrl;
      img.style.maxWidth = '240px';
      img.style.borderRadius = '8px';
      img.style.display = 'block';
      img.style.marginBottom = text ? '8px' : '0';
      bubble.appendChild(img);
    }
    if (text) {
      const p = document.createElement('p');
      p.textContent = text;
      bubble.appendChild(p);
    }
    wrap.appendChild(bubble);
    messages.appendChild(wrap);
    scrollDown();
  }

  function addAssistant() {
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = '<span class="typing"></span>';
    wrap.appendChild(bubble);
    messages.appendChild(wrap);
    scrollDown();
    return bubble;
  }

  function renderMarkdown(el, text, citations) {
    el.innerHTML = '';
    const blocks = String(text || '').split(/\n{2,}/);
    blocks.forEach((block) => {
      const lines = block.split('\n').filter((l) => l.trim());
      if (!lines.length) return;

      if (lines.every((l) => /^\s*[-*·]\s+/.test(l))) {
        const ul = document.createElement('ul');
        lines.forEach((l) => {
          const li = document.createElement('li');
          li.innerHTML = inline(l.replace(/^\s*[-*·]\s+/, ''), citations);
          ul.appendChild(li);
        });
        el.appendChild(ul);
        return;
      }

      if (/^\s*\d+\.\s+/.test(lines[0])) {
        const ol = document.createElement('ol');
        lines.forEach((l) => {
          const li = document.createElement('li');
          li.innerHTML = inline(l.replace(/^\s*\d+\.\s+/, ''), citations);
          ol.appendChild(li);
        });
        el.appendChild(ol);
        return;
      }

      const p = document.createElement('p');
      p.innerHTML = lines.map((l) => inline(l, citations)).join('<br>');
      el.appendChild(p);
    });
  }

  function inline(s, citations) {
    let out = escapeHtml(s);
    out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
    // 引用角标：[1] -> 可点击
    out = out.replace(/\[(\d{1,2})\]/g, (m, n) => {
      const idx = parseInt(n, 10);
      const exists = citations && citations.some((c) => c.index === idx);
      if (!exists) return m;
      return `<span class="cite" data-cite="${idx}">[${idx}]</span>`;
    });
    return out;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function renderCalc(bubble, results) {
    if (!results || !results.length) return;
    const box = document.createElement('div');
    box.className = 'calc-box';
    results.forEach((r) => {
      const head = document.createElement('div');
      // 单位由结果自带；卖价类工具会返回 unit='金'，
      // 中位数这类没有单位，硬拼「金」会写出「448351 金」这种错话
      const unit = r.unit ? ' ' + r.unit : '';
      head.innerHTML = `<strong>${escapeHtml(r.item || '')}</strong>　` +
        `<code>${escapeHtml(r.formula || '')}</code> = <strong>${r.value}</strong>${escapeHtml(unit)}`;
      box.appendChild(head);
      if (r.steps && r.steps.length) {
        const ul = document.createElement('ul');
        ul.className = 'steps';
        r.steps.forEach((s) => {
          const li = document.createElement('li');
          li.textContent = s;
          ul.appendChild(li);
        });
        box.appendChild(ul);
      }
    });
    bubble.appendChild(box);
  }

  function renderVision(v) {
    if (!v) return;
    const body = $('visionBody');
    body.innerHTML = '';

    const row = (label, value) => {
      if (!value) return;
      const d = document.createElement('div');
      d.className = 'item-chip';
      d.innerHTML = `<span style="color:var(--text-dim)">${escapeHtml(label)}</span>
                     <span style="margin-left:auto">${escapeHtml(String(value))}</span>`;
      body.appendChild(d);
    };

    if (v.ocr_confidence) row('OCR 置信', Number(v.ocr_confidence).toFixed(2));

    if (v.items && v.items.length) {
      v.items.forEach((it) => {
        const chip = document.createElement('div');
        chip.className = 'item-chip';
        const score = Number(it.score) || 0;
        const cls = score >= 0.82 ? 'high' : (score >= 0.55 ? 'mid' : '');
        chip.innerHTML =
          `<span>${escapeHtml(it.name || it.key || '?')}</span>
           <span class="score ${cls}">${(score * 100).toFixed(0)}%</span>`;
        body.appendChild(chip);
      });
    }

    // OCR 原文优先：这是「它到底读到了什么」的直接证据
    const texts = (v.texts || []).map((t) => String(t).trim()).filter(Boolean);
    if (texts.length) {
      const box = document.createElement('div');
      box.className = 'ocr-texts';
      texts.slice(0, 6).forEach((t) => {
        const line = document.createElement('div');
        line.textContent = t.length > 60 ? t.slice(0, 60) + '…' : t;
        box.appendChild(line);
      });
      body.appendChild(box);
    }

    const ents = (v.entities || []).filter(Boolean);
    row('识别名词', ents.slice(0, 6).join('、'));

    // 面板没有任何真实内容时，整张卡片不显示，
    // 否则用户看到空卡片会以为识别坏了
    if (body.children.length) {
      $('visionCard').hidden = false;
    }
  }

  function renderTrace(steps) {
    if (!steps || !steps.length) return;
    $('traceCard').hidden = false;
    const body = $('traceBody');
    body.innerHTML = '';
    steps.forEach((s) => {
      const div = document.createElement('div');
      div.className = 'trace-step';
      const detail = Object.entries(s)
        .filter(([k]) => !['step', 'elapsed_ms'].includes(k))
        .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`)
        .join(' ');
      div.innerHTML = `<span class="name">${escapeHtml(s.step)}</span>
                       <span class="detail">${escapeHtml(detail).slice(0, 160)}</span>`;
      body.appendChild(div);
    });
  }

  function renderCitations(citations) {
    if (!citations || !citations.length) return;
    $('citeCard').hidden = false;
    const body = $('citeBody');
    body.innerHTML = '';
    citations.forEach((c) => {
      const d = document.createElement('div');
      d.className = 'cite-item';
      d.innerHTML = `<div class="title">[${c.index}] 《${escapeHtml(c.title)}》` +
        `${c.section ? ' · ' + escapeHtml(c.section) : ''}</div>
        <div class="snippet">${escapeHtml(c.snippet || '')}</div>`;
      body.appendChild(d);
    });
  }

  function scrollDown() {
    messages.scrollTop = messages.scrollHeight;
  }

  // ------------------------------------------------------------ 交互
  function resetPanels() {
    ['visionCard', 'traceCard', 'citeCard'].forEach((id) => { $(id).hidden = true; });
  }

  async function askText(question) {
    if (busy) return;
    busy = true;
    syncSendState();
    resetPanels();
    addUser(question);
    const bubble = addAssistant();

    try {
      const r = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 6, history }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);

      bubble.innerHTML = '';
      renderMarkdown(bubble, d.answer, d.citations);
      renderCalc(bubble, d.calc);
      if (d.mocked) {
        const note = document.createElement('p');
        note.className = 'hint';
        note.textContent = '（演示模式：未配置模型 API Key）';
        bubble.appendChild(note);
      }
      renderTrace(d.trace);
      renderCitations(d.citations);
      pushHistory(question, d.answer);
    } catch (e) {
      bubble.innerHTML = `<div class="error-box">请求失败：${escapeHtml(e.message)}</div>`;
    } finally {
      busy = false;
      syncSendState();
      scrollDown();
    }
  }

  async function askVision(file, question, opts) {
    const isFollowUp = !!(opts && opts.isFollowUp);
    if (busy) return;
    busy = true;
    syncSendState();
    resetPanels();

    const url = URL.createObjectURL(file);
    addUser(question || '（请分析这张截图）', url);
    const bubble = addAssistant();

    const fd = new FormData();
    fd.append('file', file);
    if (question) fd.append('question', question);
    if (history.length) fd.append('history_json', JSON.stringify(history));

    try {
      const r = await fetch('/api/vision/ask', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);

      bubble.innerHTML = '';
      renderMarkdown(bubble, d.answer, d.citations);
      renderCalc(bubble, d.calc);
      renderVision(d.vision);
      renderTrace(d.trace);
      renderCitations(d.citations);
      pushHistory(question || d.question || '（截图提问）', d.answer);
    } catch (e) {
      bubble.innerHTML = `<div class="error-box">请求失败：${escapeHtml(e.message)}</div>`;
    } finally {
      busy = false;
      syncSendState();
      scrollDown();
      URL.revokeObjectURL(url);
    }
  }

  function submit() {
    const text = input.value.trim();
    if (!text && !pendingFile) return;
    const file = pendingFile;

    if (file) {
      rememberImage(file);
      input.value = '';
      input.style.height = 'auto';
      clearImage();
      askVision(file, text);
      return;
    }

    input.value = '';
    input.style.height = 'auto';
    clearImage();

    // 没有新图，但上一轮用过图、且这次像是在追问图片内容
    // （含"图/截图/这个/它/刚刚"等指代词）→ 自动把那张图带上
    const looksLikeImageFollowUp = /图|截图|刚才|刚刚|上面|这个|那个|它/.test(text);
    if (lastImage && looksLikeImageFollowUp && history.length) {
      askVision(lastImage.blob, text, { isFollowUp: true });
      return;
    }

    askText(text);
  }

  function clearImage() {
    pendingFile = null;
    preview.hidden = true;
    previewImg.removeAttribute('src');
    previewName.textContent = '截图';
    previewSize.textContent = '';
    fileInput.value = '';
    syncSendState();
  }

  function fmtSize(bytes) {
    if (!bytes && bytes !== 0) return '';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  }

  /* 发送按钮只在「有文字或有图」时可用。
     空输入点发送没反应，看着像坏了。 */
  function syncSendState() {
    sendBtn.disabled = busy || (!input.value.trim() && !pendingFile);
  }

  function stageFile(file) {
    pendingFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewName.textContent = file.name && file.name !== 'image.png'
      ? file.name
      : '粘贴的截图';
    previewSize.textContent = fmtSize(file.size);
    preview.hidden = false;
    syncSendState();
  }

  // ------------------------------------------------------------ 事件绑定
  sendBtn.addEventListener('click', submit);

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 130) + 'px';
    syncSendState();
  });

  fileInput.addEventListener('change', () => {
    const f = fileInput.files && fileInput.files[0];
    if (!f) return;
    if (f.size > 8 * 1024 * 1024) {
      alert('图片不能超过 8MB');
      clearImage();
      return;
    }
    stageFile(f);
  });

  removeImg.addEventListener('click', clearImage);

  document.querySelectorAll('.quick button').forEach((b) => {
    b.addEventListener('click', () => {
      input.value = b.dataset.q || '';
      submit();
    });
  });

  // 点击引用角标 → 滚动到对应来源
  messages.addEventListener('click', (e) => {
    const t = e.target.closest('.cite');
    if (!t) return;
    const idx = t.dataset.cite;
    const card = $('citeCard');
    if (card.hidden) return;
    const target = [...card.querySelectorAll('.cite-item')]
      .find((el) => el.textContent.startsWith(`[${idx}]`));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });

  // -------------------------------------------------- 贴图：支持粘贴/拖拽
  function stageImage(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) return;
    if (file.size > 8 * 1024 * 1024) {
      alert('图片不能超过 8MB');
      return;
    }
    stageFile(file);
    input.focus();
  }

  /* 记录本轮图片，供后续追问复用 */
  function rememberImage(file) {
    if (!file) return;
    if (lastImage && lastImage.url) URL.revokeObjectURL(lastImage.url);
    lastImage = { blob: file, url: URL.createObjectURL(file) };
    lastImageUsedAt = Date.now();
  }

  function extractImageFromClipboard(e) {
    const dt = e.clipboardData || window.clipboardData;
    if (!dt) return null;
    // 方式 1：items（截图粘贴走这条）
    if (dt.items && dt.items.length) {
      for (const it of dt.items) {
        if (it.kind === 'file' && it.type.startsWith('image/')) {
          return it.getAsFile();
        }
      }
    }
    // 方式 2：files（某些浏览器只给 files）
    if (dt.files && dt.files.length) {
      for (const f of dt.files) {
        if (f.type.startsWith('image/')) return f;
      }
    }
    return null;
  }

  // 粘贴：挂 document 捕获阶段，避免被输入框的默认行为拦掉
  document.addEventListener('paste', (e) => {
    const file = extractImageFromClipboard(e);
    if (file) {
      e.preventDefault();
      stageImage(file);
    }
  }, true);

  input.addEventListener('paste', (e) => {
    const file = extractImageFromClipboard(e);
    if (file) {
      e.preventDefault();
      stageImage(file);
    }
  });

  // 拖拽上传
  ['dragover', 'drop'].forEach((name) => {
    document.addEventListener(name, (e) => {
      e.preventDefault();
      if (name === 'drop') {
        const f = e.dataTransfer?.files?.[0];
        if (f) stageImage(f);
      }
    });
  });

  // 整个输入区可点击（除按钮外）
  document.querySelector('.composer')?.addEventListener('dragover', () => {
    document.querySelector('.composer').classList.add('dragover');
  });
  document.querySelector('.composer')?.addEventListener('dragleave', () => {
    document.querySelector('.composer').classList.remove('dragover');
  });

  // 初始化
  syncSendState();
  loadStatus();
  loadMeta();
  setInterval(loadStatus, 30000);
})();
