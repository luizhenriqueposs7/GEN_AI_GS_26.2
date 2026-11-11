import json

from flask import Flask, jsonify, render_template, request

from assistente import Assistente, MODELO, PASTA, carregar_corpus

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4096
app.json.ensure_ascii = False
bot = Assistente(carregar_corpus())


@app.get("/")
def chat():
    return render_template("chat.html", fontes=len(bot.docs), trechos=len(bot.trechos))


@app.post("/perguntar")
def perguntar():
    dados = request.get_json(silent=True)
    if not isinstance(dados, dict):
        return jsonify(erro="Envie uma pergunta em JSON."), 400
    pergunta, anterior = dados.get("pergunta"), dados.get("anterior")
    if not isinstance(pergunta, str) or not 1 <= len(pergunta.strip()) <= 500:
        return jsonify(erro="Escreva uma pergunta de 1 a 500 caracteres."), 400
    if anterior is not None and (
        not isinstance(anterior, str) or (anterior not in bot.docs and anterior not in bot.por_id)
    ):
        return jsonify(erro="Assunto anterior inválido. Inicie uma nova conversa."), 400
    return jsonify(bot.responder(pergunta.strip(), anterior))


@app.get("/avaliacao")
def avaliacao():
    caminho = PASTA / "resultados.json"
    resultado = json.loads(caminho.read_text(encoding="utf-8")) if caminho.is_file() else None
    return render_template("avaliacao.html", resultado=resultado, modelo=MODELO)


@app.after_request
def proteger_resposta(resposta):
    resposta.headers["Cache-Control"] = "no-store"
    resposta.headers["X-Content-Type-Options"] = "nosniff"
    resposta.headers["Content-Security-Policy"] = (
        "default-src 'self'; object-src 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    return resposta


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765)
