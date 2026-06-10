#!/usr/bin/env python3
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MagRobot Manual Control</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 28px; background: #f7f7f8; color: #1d1d1f; }
    main { max-width: 680px; margin: 0 auto; }
    h1 { font-size: 22px; margin: 0 0 16px; }
    .grid { display: grid; grid-template-columns: repeat(3, 92px); gap: 10px; margin: 18px 0; }
    .row { display: flex; gap: 10px; margin: 12px 0; flex-wrap: wrap; }
    button { height: 54px; border: 1px solid #c8c8cc; border-radius: 7px; background: white; font-size: 18px; cursor: pointer; }
    button:active { background: #dcecff; border-color: #3478f6; }
    .wide { min-width: 92px; padding: 0 18px; }
    .status { padding: 12px; background: white; border: 1px solid #ddd; border-radius: 7px; margin-top: 16px; }
    kbd { background: #eee; border: 1px solid #ccc; border-radius: 4px; padding: 2px 5px; }
  </style>
</head>
<body>
<main>
  <h1>MagRobot Manual Control</h1>
  <div class="status" id="status">Ready. Click this page, then press keys.</div>

  <div class="grid">
    <span></span><button data-key="ArrowUp">↑</button><span></span>
    <button data-key="ArrowLeft">←</button><button data-key="ArrowDown">↓</button><button data-key="ArrowRight">→</button>
  </div>

  <div class="row">
    <button class="wide" data-key="i">I pitch -</button>
    <button class="wide" data-key="k">K pitch +</button>
    <button class="wide" data-key="j">J yaw +</button>
    <button class="wide" data-key="l">L yaw -</button>
  </div>
  <div class="row">
    <button class="wide" data-key="+">+ Z</button>
    <button class="wide" data-key="-">- Z</button>
  </div>

  <p>Keyboard: <kbd>i</kbd>/<kbd>k</kbd>, <kbd>j</kbd>/<kbd>l</kbd>, arrows, <kbd>+</kbd>/<kbd>-</kbd>.</p>
</main>
<script>
let seq = 0;
async function sendKey(key) {
  seq += 1;
  const body = new URLSearchParams({key, seq: String(seq)});
  const r = await fetch('/key', {method: 'POST', body});
  document.getElementById('status').textContent = r.ok ? `Sent ${key} (#${seq})` : `Send failed: ${r.status}`;
}
document.querySelectorAll('button[data-key]').forEach(btn => {
  btn.addEventListener('click', () => sendKey(btn.dataset.key));
});
window.addEventListener('keydown', ev => {
  const allowed = ['i','j','k','l','I','J','K','L','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','+','-','='];
  if (!allowed.includes(ev.key)) return;
  ev.preventDefault();
  sendKey(ev.key);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    command_file = Path("/private/tmp/magrobot_manual_key.json")

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/key":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        params = parse_qs(self.rfile.read(length).decode("utf-8"))
        key = params.get("key", [""])[0]
        seq = params.get("seq", [str(time.time_ns())])[0]
        tmp = self.command_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "seq": seq}), encoding="utf-8")
        tmp.replace(self.command_file)
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        print("[MagRobotWeb] " + fmt % args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--command-file", default="/private/tmp/magrobot_manual_key.json")
    args = parser.parse_args()
    Handler.command_file = Path(args.command_file)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MagRobot manual control: http://127.0.0.1:{args.port}/")
    print(f"Command file: {Handler.command_file}")
    server.serve_forever()


if __name__ == "__main__":
    main()
