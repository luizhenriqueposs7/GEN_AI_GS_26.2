const form = document.querySelector("#chat-form");
const question = document.querySelector("#question");
const send = document.querySelector("#send");
const reset = document.querySelector("#reset");
const messages = document.querySelector("#messages");
const triagem = document.querySelector("#triagem");
const welcome = document.querySelector("#welcome");
const initialTriage = triagem.firstElementChild.cloneNode(true);
let assunto = null;

const labels = {
  respondido: "Orientação com fonte",
  esclarecer: "Precisamos de mais detalhes",
  fora_escopo: "Fora do escopo",
  sem_evidencia: "Evidência insuficiente",
  bloqueado: "Pedido bloqueado",
  privacidade: "Proteção de dados",
  indisponivel: "Modelo indisponível",
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function updateTriage(data) {
  triagem.replaceChildren();
  if (!data) {
    triagem.append(initialTriage.cloneNode(true));
    return;
  }
  const list = document.createElement("dl");
  for (const [label, value] of [
    ["Categoria", data.category],
    ["Prioridade sugerida", data.priority],
    ["Revisão humana", data.review_required ? "Indicada" : "Sem indicação pelas regras"],
  ]) {
    list.append(element("dt", label), element("dd", value));
  }
  triagem.append(list);
}

function appendAnswer(result) {
  const article = element("article", "", "answer " + result.status);
  article.append(element("p", labels[result.status] || "Mensagem", "answer-label"));
  article.append(element("p", result.resposta));
  if (result.fonte) {
    const a = element("a", result.fonte.id + " · " + result.fonte.titulo + " ↗", "text-link");
    a.href = result.fonte.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    article.append(a);
  }
  if (result.recuperados?.length) {
    const details = document.createElement("details");
    details.append(element("summary", "Ver evidências recuperadas (" + result.recuperados.length + ")"));
    if (result.candidatos_modelo?.length && result.candidatos_modelo.length < result.recuperados.length) {
      details.append(element("p", "Fontes compatíveis com o serviço solicitado: " + result.candidatos_modelo.join(", "), "answer-label"));
    }
    for (const trecho of result.recuperados) {
      const row = element("div", "", "evidence");
      row.append(
        element("p", trecho.id + " · " + trecho.secao, "answer-label"),
        element("p", trecho.texto),
        element("small", "RRF " + trecho.score.toFixed(4) + " · Consulta " + trecho.consulta),
      );
      details.append(row);
    }
    article.append(details);
  }
  messages.append(article);
  article.scrollIntoView({block: "nearest"});
  updateTriage(result.triagem);
  if (result.assunto) assunto = result.assunto;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = question.value.trim();
  if (!input || send.disabled) return;
  welcome.hidden = true;
  // Entradas com dados pessoais ficam fora do histórico visível.
  const bubble = element("p", "Pergunta enviada", "bubble");
  const waiting = element("p", "Consultando a base de zeladoria…");
  messages.append(bubble, waiting);
  waiting.scrollIntoView({block: "nearest"});
  question.value = "";
  send.disabled = reset.disabled = true;
  try {
    const response = await fetch("/perguntar", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({pergunta: input, anterior: assunto}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.erro);
    bubble.textContent = result.status === "privacidade" ? "Pergunta omitida por privacidade" : input;
    waiting.remove();
    appendAnswer(result);
  } catch {
    waiting.remove();
    appendAnswer({status: "indisponivel", resposta: "Não foi possível consultar o assistente. Tente novamente."});
  } finally {
    send.disabled = reset.disabled = false;
    question.focus({preventScroll: true});
  }
});

reset.addEventListener("click", () => {
  assunto = null;
  messages.replaceChildren(welcome);
  welcome.hidden = false;
  question.value = "";
  updateTriage(null);
  question.focus();
});

for (const button of document.querySelectorAll(".examples button")) {
  button.addEventListener("click", () => {
    question.value = button.textContent;
    question.focus();
  });
}
