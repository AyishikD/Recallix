/* The public demo: the browser talks only to the Cloudflare Worker (deploy/cloudflare/demo-gate), which checks
   the visitor is human (Turnstile), applies the limits and forwards to the demo server on Modal. */
export const DEMO_API = "https://demo-api.recallic.com";
/* Turnstile site key: public by design (it is in every page that shows the widget). The secret key is only in the Worker. */
export const TURNSTILE_SITE_KEY = "0x4AAAAAAGXIJqMLDnw5vLYf";
