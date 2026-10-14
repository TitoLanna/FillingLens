/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the RAG API. Leave empty to use the dev proxy / same origin. */
  readonly VITE_API_BASE_URL?: string;
  /** "false" to call the real API; anything else uses the in-browser mock. */
  readonly VITE_USE_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
