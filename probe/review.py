"""A localhost-only form for the two required human reviews."""
import html
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from .common import read_jsonl, require, write_jsonl


def text_block(label, text):
    return f"<h3>{html.escape(label)}</h3><pre>{html.escape(text)}</pre>"


def decision_select(name, value, label):
    options = [("", "Pending"), ("true", "Yes"), ("false", "No")]
    selected = "" if value is None else str(value).lower()
    return f'<label>{html.escape(label)} <select name="{name}">' + "".join(
        f'<option value="{v}" {"selected" if selected == v else ""}>{t}</option>' for v, t in options) + "</select></label>"


def serve(root, kind, port):
    path = root / "review" / ("negatives.jsonl" if kind == "negatives" else "generation.jsonl")
    require(path.exists(), f"Prepare {kind} review first")
    token = secrets.token_urlsafe(24)
    test = {r["id"]: r for r in read_jsonl(root / "data" / "test.jsonl")}
    candidates = {r["id"]: r for r in read_jsonl(root / "data" / "candidates.jsonl")}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/":
                return self.reply(404, "Not found")
            rows = read_jsonl(path)
            try:
                index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                require(0 <= index < len(rows), "Invalid review index")
            except (ValueError, IndexError):
                return self.reply(400, "Invalid review index")
            r = rows[index]
            flags = ["gold_valid", "easy_valid", "hard_valid"] if kind == "negatives" else ["addresses_task", "substantially_correct"]
            reviewed = sum(all(type(x.get(k)) is bool for k in flags) and bool(x.get("adjudicator", "").strip()) for x in rows)
            body = ["<!doctype html><meta charset='utf-8'><title>GPT-2 study review</title>",
                    "<style>body{font:17px system-ui;max-width:900px;margin:32px auto;padding:0 20px;background:#f7f8fa;color:#172033}pre{white-space:pre-wrap;background:white;padding:18px;border:1px solid #dce1e8;border-radius:8px;font:16px/1.5 system-ui}label{display:block;margin:16px 0}input,select,textarea,button{font:inherit;padding:8px}textarea{width:95%;height:80px}nav{display:flex;gap:24px;margin:24px 0}button{background:#174f80;color:white;border:0;border-radius:6px;cursor:pointer}details{margin:20px 0}</style>",
                    f"<h1>{'Negative validation' if kind == 'negatives' else 'Blinded answer grading'}</h1>",
                    f"<p>Item {index+1} of {len(rows)} · {reviewed} reviewed. Decisions save to the local experiment.</p>",
                    f'<nav><a href="/?index={max(0,index-1)}">Previous</a><a href="/?index={min(len(rows)-1,index+1)}">Next without saving</a></nav>',
                    f'<form method="post"><input type="hidden" name="token" value="{token}"><input type="hidden" name="index" value="{index}">']
            if kind == "negatives":
                target = test[r["id"]]
                body += [f"<p>{html.escape(r['id'])} · {html.escape(target['category'])}</p>",
                         "<p>Read the prompt and full context. Validate the gold answer. Easy and hard answers must be incorrect; the hard answer must also be related and plausible. Do not use model scores to choose a negative.</p>",
                         text_block("Instruction", target["instruction"]), text_block("Context", target["context"] or "(none)"),
                         text_block("Gold response", target["response"]), decision_select("gold_valid", r["gold_valid"], "Gold response acceptable?"),
                         text_block("Easy negative", test[r["easy_id"]]["response"]), decision_select("easy_valid", r["easy_valid"], "Easy response is incorrect?")]
                body.append('<label>Hard negative choice <select name="hard_id">')
                for option in candidates[r["id"]]["hard_options"]:
                    selected = "selected" if option["id"] == r["hard_id"] else ""
                    body.append(f'<option {selected} value="{option["id"]}">{option["id"]} (similarity {option["similarity"]:.2f})</option>')
                body.append("</select></label>")
                for option in candidates[r["id"]]["hard_options"]:
                    source = test[option["id"]]
                    body += [f"<details open><summary>{html.escape(option['id'])} · length ratio {option['length_ratio']:.2f}</summary>",
                             text_block("Candidate answer", source["response"]), text_block("Its original instruction", source["instruction"]), "</details>"]
                body.append(decision_select("hard_valid", r["hard_valid"], "Chosen hard response is related, plausible, and incorrect?"))
            else:
                body += [f"<p>{html.escape(r['blind_id'])}</p>",
                         "<p>Task adherence requires performing the requested operation and its essential constraints. Correctness requires a coherent, substantially correct answer. Accept valid alternatives to the reference. Empty answers fail; judge truncated answers as shown.</p>",
                         text_block("Instruction", r["instruction"]), text_block("Context", r["context"] or "(none)"),
                         text_block("Reference (not necessarily the only correct answer)", r["reference"]), text_block("Anonymous model answer", r["answer"]),
                         decision_select("addresses_task", r["addresses_task"], "Addresses the requested task?"),
                         decision_select("substantially_correct", r["substantially_correct"], "Substantially correct?")]
            body += [f'<label>Adjudicator <input id="reviewer" name="reviewer" value="{html.escape(r.get("adjudicator", r.get("reviewer", "")),quote=True)}"></label>',
                     '<label>Type <select name="adjudicator_type"><option value="human">Human</option><option value="llm">LLM</option></select></label>',
                     f'<label>Reason or uncertainty <textarea name="notes">{html.escape(r["notes"])}</textarea></label>',
                     '<button type="submit">Save and next</button></form>',
                     "<script>const r=document.getElementById('reviewer');if(!r.value)r.value=localStorage.getItem('reviewer')||'';r.addEventListener('change',()=>localStorage.setItem('reviewer',r.value));</script>"]
            self.reply(200, "\n".join(body))

        def do_POST(self):
            try:
                require(urlparse(self.path).path == "/", "Invalid path")
                length = int(self.headers.get("Content-Length", "0"))
                require(0 < length <= 65536, "Invalid request length")
                values = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
                require(secrets.compare_digest(values.get("token", [""])[0], token), "Invalid form token")
                rows = read_jsonl(path)
                index = int(values["index"][0])
                require(0 <= index < len(rows), "Invalid review index")
                r = rows[index]
                flags = ["gold_valid", "easy_valid", "hard_valid"] if kind == "negatives" else ["addresses_task", "substantially_correct"]
                for flag in flags:
                    value = values[flag][0]
                    require(value in {"", "true", "false"}, "Invalid decision")
                    r[flag] = {"": None, "true": True, "false": False}[value]
                if kind == "negatives":
                    hard_id = values["hard_id"][0]
                    require(hard_id in {x["id"] for x in candidates[r["id"]]["hard_options"]}, "Invalid hard candidate")
                    r["hard_id"] = hard_id
                r["adjudicator"] = values.get("reviewer", [""])[0].strip()
                r["adjudicator_type"] = values.get("adjudicator_type", ["human"])[0]
                r["notes"] = values.get("notes", [""])[0]
                write_jsonl(path, rows)
                self.send_response(303)
                self.send_header("Location", f"/?index={min(index+1,len(rows)-1)}")
                self.end_headers()
            except (ValueError, KeyError, IndexError) as error:
                self.reply(400, html.escape(str(error)))

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Human review: http://127.0.0.1:{port} (Ctrl-C stops the server)", flush=True)
    server.serve_forever()
