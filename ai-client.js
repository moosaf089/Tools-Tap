/* ToolsTap AI connector for a public Hugging Face Gradio/ZeroGPU Space.
 * No private HF token belongs in this public website code.
 */
(function (global) {
  'use strict';

  const CLIENT_CDN = 'https://cdn.jsdelivr.net/npm/@gradio/client@2.7.1/dist/index.min.js';
  const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
  let clientBundlePromise = null;
  let clientPromise = null;

  function baseUrl() {
    return String(global.TOOLSTAP_AI_API_BASE || '').trim().replace(/\/+$/, '');
  }

  function isConfigured() {
    return /^https:\/\//i.test(baseUrl());
  }

  function url(path) {
    if (!path) return baseUrl();
    if (/^https?:\/\//i.test(String(path))) return String(path);
    if (!isConfigured()) throw new Error('ToolsTap AI server is not configured.');
    return new URL(String(path).startsWith('/') ? String(path) : '/' + path, baseUrl()).href;
  }

  function friendlyError(error) {
    const raw = error && (error.message || error.toString) ? String(error.message || error) : 'Unknown error';
    const lower = raw.toLowerCase();
    if (lower.includes('quota') || lower.includes('daily limit') || lower.includes('gpu hours')) {
      return 'Hugging Face ZeroGPU ki daily free GPU limit khatam ho sakti hai. Baad mein retry karein; free GPU unlimited nahi hai.';
    }
    if (lower.includes('not running') || lower.includes('sleep') || lower.includes('building') || lower.includes('connect') || lower.includes('failed to fetch') || lower.includes('network')) {
      return 'ToolsTap AI server abhi start nahi hua ya connection nahi ban saka. Hugging Face Space ka status Running check karein, phir dobara try karein. Detail: ' + raw.slice(0, 300);
    }
    if (lower.includes('out of memory') || lower.includes('cuda out of memory') || lower.includes('memory')) {
      return 'AI server ki memory limit hit hui. Chhoti source image ya kam output scale try karein. Detail: ' + raw.slice(0, 300);
    }
    return raw.slice(0, 500) || 'AI processing failed. Space logs check karein.';
  }

  async function getClientBundle() {
    if (clientBundlePromise) return clientBundlePromise;
    clientBundlePromise = import(CLIENT_CDN).then((mod) => {
      if (!mod || typeof mod.Client?.connect !== 'function' || typeof mod.handle_file !== 'function') {
        throw new Error('Gradio JavaScript client loaded but required functions were missing.');
      }
      return mod;
    }).catch((error) => {
      clientBundlePromise = null;
      throw error;
    });
    return clientBundlePromise;
  }

  async function getClient(onProgress) {
    if (!isConfigured()) throw new Error('ToolsTap AI backend URL abhi configure nahi hai. ai-config.js mein live Space URL set karein.');
    if (clientPromise) return clientPromise;
    clientPromise = (async () => {
      onProgress && onProgress('Gradio client load ho raha hai…');
      const { Client, handle_file } = await getClientBundle();
      onProgress && onProgress('Hugging Face AI Space se connection ban raha hai…');
      const client = await Client.connect(baseUrl(), {
        events: ['data', 'status'],
        status_callback: (status) => {
          if (!onProgress || !status) return;
          const state = status.status || status.detail || '';
          if (state === 'building' || state === 'BUILDING') onProgress('AI Space abhi build ho raha hai. Build finish hone ka intezar karein…');
          else if (state === 'sleeping' || state === 'SLEEPING') onProgress('AI Space wake up ho raha hai. Pehli request mein waqt lag sakta hai…');
          else if (state === 'error' || state === 'space_error') onProgress('AI Space error state mein hai. Hugging Face Logs check karna hoga.');
        }
      });
      return { client, handle_file };
    })().catch((error) => {
      clientPromise = null;
      throw new Error(friendlyError(error));
    });
    return clientPromise;
  }

  function endpointFor(path) {
    const normalized = String(path || '').toLowerCase();
    if (normalized.includes('remove-background')) return '/remove_background';
    if (normalized.includes('line-art-upscale')) return '/line_art_upscale';
    if (normalized.includes('/upscale')) return '/upscale_image';
    throw new Error('Unknown ToolsTap AI operation: ' + path);
  }

  function fileUrl(ref) {
    if (!ref) throw new Error('AI server did not return a file for preview/download. Check Space logs.');
    if (typeof ref === 'string') {
      if (/^https?:\/\//i.test(ref)) return ref;
      if (!isConfigured()) throw new Error('AI server URL is missing.');
      if (ref.startsWith('/')) return new URL(ref, baseUrl()).href;
      return new URL('/gradio_api/file=' + encodeURI(ref), baseUrl()).href;
    }
    if (ref.url) {
      if (/^https?:\/\//i.test(ref.url)) return ref.url;
      return new URL(ref.url, baseUrl()).href;
    }
    if (ref.path) return new URL('/gradio_api/file=' + encodeURI(ref.path), baseUrl()).href;
    throw new Error('AI server returned an unknown file format. Open the Space API page to verify the outputs.');
  }

  function normalizeResult(data, sourceFile) {
    if (!Array.isArray(data) || data.length < 2) {
      throw new Error('AI Space response format did not match expected preview, download and metadata outputs. Check the app API and rebuild logs.');
    }
    const preview = data[0];
    const download = data[1];
    const meta = data[2] && typeof data[2] === 'object' ? data[2] : {};
    const downloadName = meta.filename || download.orig_name || download.name || ((sourceFile && sourceFile.name) || 'toolstap-image.png');
    return {
      id: null,
      preview_url: fileUrl(preview),
      download_url: fileUrl(download),
      filename: downloadName,
      width: Number(meta.width || download.width || 0),
      height: Number(meta.height || download.height || 0),
      size_bytes: Number(meta.size_bytes || download.size || 0),
      mime_type: String(meta.mime_type || download.mime_type || 'application/octet-stream'),
      note: meta.note || 'Preview is downsized; download contains the full-size output.'
    };
  }

  async function processImage(path, file, fields, onProgress) {
    if (!file) throw new Error('Pehle image upload karein.');
    if (file.size > MAX_UPLOAD_BYTES) throw new Error('Image 25 MB se bari hai. Chhoti image select karein.');
    if (!isConfigured()) throw new Error('ToolsTap AI backend configure nahi hai. ai-config.js mein Hugging Face Space ka URL set karein.');
    const input = fields || {};
    const operation = endpointFor(path);
    const { client, handle_file } = await getClient(onProgress);
    const fileInput = handle_file(file);
    let payload;

    if (operation === '/remove_background') {
      payload = [fileInput, String(input.model || 'fast'), String(input.background || 'transparent'), String(input.output_format || 'png'), String(input.filename || file.name || 'image.png')];
    } else if (operation === '/upscale_image') {
      payload = [fileInput, String(input.scale || '2'), String(input.model || 'general'), String(input.output_format || 'png'), String(input.filename || file.name || 'image.png')];
    } else {
      payload = [
        fileInput,
        String(input.scale || '2'),
        String(input.output_long_edge || '0'),
        String(input.edge_mode || 'balanced'),
        Number(input.sharpness ?? 0.45),
        String(input.bw_cleanup).toLowerCase() === 'true' || input.bw_cleanup === true,
        Number(input.threshold ?? 170),
        String(input.use_ai).toLowerCase() !== 'false' && input.use_ai !== false,
        String(input.filename || file.name || 'line-art.png')
      ];
    }

    onProgress && onProgress('Image securely upload ho rahi hai. Original aapke device par rehti hai jab tak aap submit na karein.');
    let submit;
    try {
      submit = client.submit(operation, payload);
    } catch (error) {
      throw new Error(friendlyError(error));
    }

    let responseData = null;
    let lastStatus = '';
    const started = Date.now();
    try {
      for await (const message of submit) {
        if (message && message.type === 'status') {
          const status = message.status || message;
          if (status.stage === 'error' || status.success === false) {
            throw new Error(status.message || status.code || 'Hugging Face Space ne processing reject kar di.');
          }
          if (status.stage === 'pending') {
            const pos = Number.isFinite(status.position) ? ` Queue position: ${status.position + 1}.` : '';
            lastStatus = 'ZeroGPU queue mein intezar ho raha hai.' + pos;
            onProgress && onProgress(lastStatus);
          } else if (status.stage === 'generating') {
            lastStatus = 'AI model server par image process kar raha hai…';
            onProgress && onProgress(lastStatus);
          } else if (status.stage === 'complete') {
            onProgress && onProgress('AI result tayyar hai; preview aur download finalize ho rahe hain…');
          } else if (status.message) {
            onProgress && onProgress(String(status.message).slice(0, 200));
          }
        } else if (message && message.type === 'data') {
          responseData = message.data;
        }
        if (Date.now() - started > 20 * 60 * 1000) {
          if (typeof submit.cancel === 'function') submit.cancel();
          throw new Error('AI request ne 20 minutes se zyada liye. Space status, ZeroGPU quota aur server logs check karein.');
        }
      }
    } catch (error) {
      throw new Error(friendlyError(error));
    }
    if (!responseData) throw new Error('AI Space ne result return nahi kiya. Space Running status, Use via API endpoints aur Logs check karein.');
    try {
      const normalized = normalizeResult(responseData, file);
      onProgress && onProgress('Processing complete. Full-resolution result download ke liye ready hai.');
      return normalized;
    } catch (error) {
      throw new Error(friendlyError(error));
    }
  }

  async function deleteResult() {
    // Gradio manages temporary output cache and expires cached files on the server.
    return undefined;
  }

  global.ToolsTapAI = { isConfigured, baseUrl, url, processImage, deleteResult };
})(window);
