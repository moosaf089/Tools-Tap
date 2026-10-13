/* Shared client for ToolsTap server-side AI processing. No secret API key is stored here. */
(function (global) {
  'use strict';
  function baseUrl() {
    return String(global.TOOLSTAP_AI_API_BASE || '').trim().replace(/\/+$/, '');
  }
  function isConfigured() {
    const base = baseUrl();
    return /^https:\/\//i.test(base) || /^https?:\/\/localhost(?::\d+)?$/i.test(base) || /^https?:\/\/127\.0\.0\.1(?::\d+)?$/i.test(base);
  }
  function url(path) {
    if (!isConfigured()) throw new Error('ToolsTap AI server is not configured yet.');
    return baseUrl() + (path.startsWith('/') ? path : '/' + path);
  }
  async function readError(response) {
    let message = `AI server returned HTTP ${response.status}.`;
    try {
      const body = await response.json();
      message = body.detail || body.error || body.message || message;
    } catch (_) {
      try { const body = await response.text(); if (body) message = body.slice(0, 500); } catch (_) {}
    }
    return String(message);
  }
  async function processImage(path, file, fields, onProgress) {
    if (file && file.size > 25 * 1024 * 1024) throw new Error('This upload exceeds the AI backend 25 MB limit. Choose a smaller image.');
    if (!isConfigured()) throw new Error('ToolsTap AI server is not configured. Using safe browser mode for smaller images.');
    const form = new FormData();
    form.append('file', file, file.name || 'image.png');
    Object.entries(fields || {}).forEach(([key, value]) => {
      if (value !== undefined && value !== null) form.append(key, String(value));
    });
    onProgress && onProgress('Uploading image to the ToolsTap AI worker…');
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 180000);
    let job;
    try {
      const response = await fetch(url(path), { method: 'POST', body: form, signal: controller.signal, cache: 'no-store' });
      if (!response.ok) throw new Error(await readError(response));
      job = await response.json();
    } catch (error) {
      if (error && error.name === 'AbortError') throw new Error('Upload timed out. Check your connection and retry; your original image is still on this device.');
      if (error instanceof TypeError) throw new Error('Could not reach the ToolsTap AI server. Check its status and CORS settings, then retry.');
      throw error;
    } finally { clearTimeout(timer); }

    if (!job || !job.job_id) {
      if (job && job.id && job.preview_url && job.download_url) return job;
      throw new Error('AI server did not return a valid job ID. Check that the backend deployment is current.');
    }
    const started = Date.now();
    while (Date.now() - started < 20 * 60 * 1000) {
      await new Promise(resolve => setTimeout(resolve, 1600));
      let response;
      try { response = await fetch(url('/api/jobs/' + encodeURIComponent(job.job_id)), { cache: 'no-store' }); }
      catch (_) { throw new Error('Connection to the AI worker was lost while processing. Wait a moment and try again.'); }
      if (!response.ok) throw new Error(await readError(response));
      const state = await response.json();
      if (onProgress && state.message) onProgress(state.message);
      if (state.status === 'complete' && state.result) return state.result;
      if (state.status === 'error') throw new Error(state.error || 'The AI worker could not process this image.');
    }
    throw new Error('This image took longer than 20 minutes. Try a smaller image or a more powerful server.');
  }
  async function deleteResult(id) {
    if (!id || !isConfigured()) return;
    try { await fetch(url('/api/results/' + encodeURIComponent(id)), { method: 'DELETE', cache: 'no-store' }); } catch (_) {}
  }
  global.ToolsTapAI = { isConfigured, baseUrl, url, processImage, deleteResult };
})(window);
