// vision.js — client-side image diagnosis via Google Gemini Vision API.
//
// This call happens entirely in the browser, not on the Flask server.
// Reason: this app's backend runs in a sandboxed environment whose outbound
// network is restricted to a fixed allowlist that does not include Google's
// API host. Running the Gemini call client-side also means the user's API
// key never has to pass through (or be stored on) the server.
//
// The user supplies their own Gemini API key (stored only in
// localStorage on their own machine — never sent to our Flask backend).

const GEMINI_MODEL = 'gemini-2.0-flash';
const GEMINI_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

const PUTER_VISION_PROMPT = `You are looking at a photo related to a computer hardware or software problem.
Describe in 1-2 plain sentences what you observe that is relevant to diagnosing a fault
(e.g. visible damage, cable state, an error message on screen, indicator lights, dust,
physical condition of a component, BIOS/UEFI screen, Windows recovery screen, app error).
Be specific and factual.

Then classify the most likely fault category from EXACTLY this list:
power_supply_failure, overheating, ram_failure, storage_failure, display_gpu_failure,
peripheral_issue, boot_recovery_issue, bios_firmware_issue, driver_issue,
application_issue, unclear

Respond with ONLY raw JSON, no markdown formatting, no backticks, in this exact shape:
{"description": "...", "suspected_category": "..."}`;

const VALID_CATEGORIES = [
  'power_supply_failure',
  'overheating',
  'ram_failure',
  'storage_failure',
  'display_gpu_failure',
  'peripheral_issue',
  'boot_recovery_issue',
  'bios_firmware_issue',
  'driver_issue',
  'application_issue',
];

function getGeminiApiKey() {
  return localStorage.getItem('gemini_api_key') || '';
}

function setGeminiApiKey(key) {
  localStorage.setItem('gemini_api_key', key.trim());
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // strip the "data:image/png;base64," prefix
      const result = reader.result;
      const base64 = result.substring(result.indexOf(',') + 1);
      resolve(base64);
    };
    reader.onerror = () => reject(new Error('Could not read the image file.'));
    reader.readAsDataURL(file);
  });
}

const VISION_PROMPT = `You are looking at a photo related to a computer hardware or software problem.
Describe in 1-2 plain sentences what you observe that is relevant to diagnosing a fault
(e.g. visible damage, cable state, an error message on screen, indicator lights, dust,
physical condition of a component, BIOS/UEFI screen, Windows recovery screen, app error).
Be specific and factual; do not guess at things you cannot actually see.

Then classify the most likely fault category from EXACTLY this list:
power_supply_failure, overheating, ram_failure, storage_failure, display_gpu_failure,
peripheral_issue, boot_recovery_issue, bios_firmware_issue, driver_issue,
application_issue, unclear

Respond with ONLY raw JSON, no markdown formatting, no backticks, in this exact shape:
{"description": "...", "suspected_category": "..."}`;

/**
 * Sends an image to Gemini Vision and returns a finding object:
 *   { description: string, suspected_category: string|null }
 * Throws an Error with a user-friendly message on failure.
 */
async function analyzeImageWithGemini(file) {
  const apiKey = getGeminiApiKey();
  if (!apiKey) {
    throw new Error('NO_API_KEY');
  }

  if (file.size > 15 * 1024 * 1024) {
    throw new Error('Image is too large (max 15MB). Please use a smaller photo.');
  }

  const base64Data = await fileToBase64(file);

  const body = {
    contents: [
      {
        parts: [
          { text: VISION_PROMPT },
          { inline_data: { mime_type: file.type || 'image/jpeg', data: base64Data } },
        ],
      },
    ],
    generationConfig: {
      temperature: 0.2,
      response_mime_type: 'application/json',
    },
  };

  const MAX_RETRIES = 3;
  let lastErr;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    let res;
    try {
      res = await fetch(GEMINI_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
        },
        body: JSON.stringify(body),
      });
    } catch (networkErr) {
      lastErr = new Error('Could not reach Gemini. Check your internet connection.');
      continue;
    }

    if (res.ok) {
      const data = await res.json();
      const candidate = data?.candidates?.[0];
      const textPart = candidate?.content?.parts?.find(p => p.text)?.text;

      if (!textPart) {
        throw new Error('Gemini did not return a usable response for this image.');
      }

      let parsed;
      try {
        const cleaned = textPart.replace(/```json|```/g, '').trim();
        parsed = JSON.parse(cleaned);
      } catch (parseErr) {
        return { description: textPart.trim(), suspected_category: null };
      }

      const category = VALID_CATEGORIES.includes(parsed.suspected_category)
        ? parsed.suspected_category
        : null;

      const normalizedDescription = (parsed.description || textPart.trim()).toLowerCase();
      const bootHints = [
        'hp sure recover',
        'no operating system was found',
        'operating system not found',
        'boot device',
        'startup repair',
        'restore from network',
        'restore from local drive',
        'content key',
        'press f11',
        'press esc',
        'recovery screen',
      ];
      const biosHints = ['bios', 'uefi', 'firmware', 'secure boot', 'boot order'];
      const driverHints = ['driver', 'device manager', 'rollback driver'];
      const appHints = ['application', 'app', 'not responding', 'crash', 'freeze', 'software'];

      let inferredCategory = category;
      if (bootHints.some(hint => normalizedDescription.includes(hint))) {
        inferredCategory = 'boot_recovery_issue';
      } else if (biosHints.some(hint => normalizedDescription.includes(hint))) {
        inferredCategory = 'bios_firmware_issue';
      } else if (driverHints.some(hint => normalizedDescription.includes(hint))) {
        inferredCategory = 'driver_issue';
      } else if (appHints.some(hint => normalizedDescription.includes(hint))) {
        inferredCategory = 'application_issue';
      }

      return {
        description: parsed.description || textPart.trim(),
        suspected_category: inferredCategory,
      };
    }

    if (res.status === 429) {
      const retryAfter = parseInt(res.headers.get('retry-after') || '2', 10);
      const waitMs = Math.min(retryAfter * 1000 * Math.pow(2, attempt), 16000);
      lastErr = new Error('Gemini rate limit reached — please wait a moment and try again.');
      await new Promise(r => setTimeout(r, waitMs));
      continue;
    }

    if (res.status === 400 || res.status === 403) {
      throw new Error('INVALID_API_KEY');
    }
    throw new Error(`Gemini request failed (HTTP ${res.status}).`);
  }

  throw lastErr;
}

/**
 * Sends an image to Puter.js AI chat (free serverless fallback) and returns
 * a finding object: { description: string, suspected_category: string|null }
 * Throws an Error on failure.
 */
async function analyzeImageWithPuter(file) {
  if (typeof puter === 'undefined' || !puter.ai || !puter.ai.chat) {
    throw new Error('Puter.js not available');
  }

  if (file.size > 10 * 1024 * 1024) {
    throw new Error('Image is too large for Puter.js (max 10MB).');
  }

  const resp = await puter.ai.chat(
    PUTER_VISION_PROMPT,
    file,
    { model: 'gpt-5.4-nano' }
  );

  const text = resp?.message?.content;
  if (!text || typeof text !== 'string') {
    throw new Error('Puter.js did not return a usable response for this image.');
  }

  let parsed;
  try {
    const cleaned = text.replace(/```json|```/g, '').trim();
    parsed = JSON.parse(cleaned);
  } catch (parseErr) {
    return { description: text.trim(), suspected_category: null };
  }

  const category = VALID_CATEGORIES.includes(parsed.suspected_category)
    ? parsed.suspected_category
    : null;

  const normalizedDescription = (parsed.description || text.trim()).toLowerCase();
  const bootHints = [
    'hp sure recover',
    'no operating system was found',
    'operating system not found',
    'boot device',
    'startup repair',
    'restore from network',
    'restore from local drive',
    'content key',
    'press f11',
    'press esc',
    'recovery screen',
  ];
  const biosHints = ['bios', 'uefi', 'firmware', 'secure boot', 'boot order'];
  const driverHints = ['driver', 'device manager', 'rollback driver'];
  const appHints = ['application', 'app', 'not responding', 'crash', 'freeze', 'software'];

  let inferredCategory = category;
  if (bootHints.some(hint => normalizedDescription.includes(hint))) {
    inferredCategory = 'boot_recovery_issue';
  } else if (biosHints.some(hint => normalizedDescription.includes(hint))) {
    inferredCategory = 'bios_firmware_issue';
  } else if (driverHints.some(hint => normalizedDescription.includes(hint))) {
    inferredCategory = 'driver_issue';
  } else if (appHints.some(hint => normalizedDescription.includes(hint))) {
    inferredCategory = 'application_issue';
  }

  return {
    description: parsed.description || text.trim(),
    suspected_category: inferredCategory,
  };
}
