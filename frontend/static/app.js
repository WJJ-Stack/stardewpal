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
  const removeImg = $('removeImg');

  let pendingFile = null;
  let busy = false;

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
      head.innerHTML = `<strong>${escapeHtml(r.item || '')}</strong>　` +
        `<code>${escapeHtml(r.formula || '')}</code> = <strong>${r.value}</strong> 金`;
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
    $('visionCard').hidden = false;
    const body = $('visionBody');
    body.innerHTML = '';

    const screenMap = {
      inventory: '背包', shop: '商店', fishing: '钓鱼', dialogue: '对话',
      quest: '任务', farm: '农场', cooking: '烹饪', other: '其他', unknown: '未识别',
    };
    const row = (k, val) => {
      const d = document.createElement('div');
      d.className = 'item-chip';
      d.innerHTML = `<span style="color:var(--text-dim)">${k}</span>
                     <span style="margin-left:auto">${escapeHtml(String(val))}</span>`;
      body.appendChild(d);
    };
    row('界面', screenMap[v.screen] || v.screen || '未识别');
    if (v.ocr_confidence) row('OCR 置信', v.ocr_confidence.toFixed(2));

    if (v.items && v.items.length) {
      v.items.forEach((it) => {
        const chip = document.createElement('div');
        chip.className = 'item-chip';
        const cls = it.score >= 0.82 ? 'high' : (it.score >= 0.55 ? 'mid' : '');
        chip.innerHTML =
          `<span>${escapeHtml(it.name || it.key || '?')}</span>
           <span class="score ${cls}">${(it.score * 100).toFixed(0)}%</span>`;
        body.appendChild(chip);
      });
    }
    if (v.entities && v.entities.length) {
      row('识别文字', v.entities.slice(0, 5).join('、'));
    }
    if (v.needs_confirm) {
      const warn = document.createElement('div');
      warn.className = 'error-box';
      warn.style.marginTop = '8px';
      warn.textContent = '识别置信度较低，结果仅供参考，可补充文字提问以确认。';
      body.appendChild(warn);
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
    sendBtn.disabled = true;
    resetPanels();
    addUser(question);
    const bubble = addAssistant();

    try {
      const r = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 6 }),
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
    } catch (e) {
      bubble.innerHTML = `<div class="error-box">请求失败：${escapeHtml(e.message)}</div>`;
    } finally {
      busy = false;
      sendBtn.disabled = false;
      scrollDown();
    }
  }

  async function askVision(file, question) {
    if (busy) return;
    busy = true;
    sendBtn.disabled = true;
    resetPanels();

    const url = URL.createObjectURL(file);
    addUser(question || '（请分析这张截图）', url);
    const bubble = addAssistant();

    const fd = new FormData();
    fd.append('file', file);
    if (question) fd.append('question', question);

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
    } catch (e) {
      bubble.innerHTML = `<div class="error-box">请求失败：${escapeHtml(e.message)}</div>`;
    } finally {
      busy = false;
      sendBtn.disabled = false;
      scrollDown();
      URL.revokeObjectURL(url);
    }
  }

  function submit() {
    const text = input.value.trim();
    if (!text && !pendingFile) return;
    const file = pendingFile;
    clearImage();
    input.value = '';
    input.style.height = 'auto';
    if (file) {
      askVision(file, text);
    } else {
      askText(text);
    }
  }

  function clearImage() {
    pendingFile = null;
    preview.hidden = true;
    previewImg.removeAttribute('src');
    fileInput.value = '';
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
  });

  fileInput.addEventListener('change', () => {
    const f = fileInput.files && fileInput.files[0];
    if (!f) return;
    if (f.size > 8 * 1024 * 1024) {
      alert('图片不能超过 8MB');
      clearImage();
      return;
    }
    pendingFile = f;
    previewImg.src = URL.createObjectURL(f);
    preview.hidden = false;
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

  // 粘贴图片
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])]
      .find((i) => i.type.startsWith('image/'));
    if (!item) return;
    const f = item.getAsFile();
    if (f) {
      pendingFile = f;
      previewImg.src = URL.createObjectURL(f);
      preview.hidden = false;
    }
  });

  // 初始化
  loadStatus();
  loadMeta();
  setInterval(loadStatus, 30000);
})();
