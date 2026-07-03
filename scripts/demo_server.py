"""Tiny local demo: a one-page UI for querying the fine-tuned model by hand.

Usage:
    python scripts/demo_server.py        # then open http://localhost:8765

Loads the base model with the trained LoRA adapter once at startup, serves
demo/index.html, and answers POST /generate with {"schema", "question"} ->
{"sql"}. Reuses the exact prompt template and SQL extraction used in training
and evaluation, so what you test here is what was measured.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from mlx_lm import generate, load

from generate_predictions import extract_sql
from prompt_format import build_messages

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "demo" / "index.html").read_bytes()
PORT = 8765

print("Loading model + adapter...")
MODEL, TOKENIZER = load(
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit", adapter_path=str(ROOT / "adapters")
)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = TOKENIZER.apply_chat_template(
            build_messages(body["question"], body["schema"]), add_generation_prompt=True
        )
        sql = extract_sql(generate(MODEL, TOKENIZER, prompt=prompt, max_tokens=120))
        payload = json.dumps({"sql": sql}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Demo running at http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
