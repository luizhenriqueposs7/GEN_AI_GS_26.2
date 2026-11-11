# CitizenTech · Assistente de zeladoria urbana

**FIAP · Global Solution 2026.2 · Generative AI & Advanced Nets · 2TIAPF**

- Luiz Henrique Poss · RM562177
- Luis Filipe Chaves Amorim · RM564706
- Felipe Stefani Honorato · RM563380

## Links da entrega

- [Demonstração no YouTube](https://youtu.be/BDX_AxOmyHw)
- [Repositório informado no GitHub](https://github.com/luizhenriqueposs7/GEN_AI_GS_26.2)

O assistente orienta sobre serviços de zeladoria do município de São Paulo. Reaproveita as regras de categoria, prioridade e revisão do notebook CitizenTech e consulta sete páginas municipais, organizadas em 20 trechos. A interface exibe a orientação, a fonte oficial e as evidências recuperadas.

## Executar no Windows

Use Windows x64, Python 3.13 e PowerShell. Abra o PowerShell na pasta deste README e execute os comandos abaixo. Um ambiente virtual mantém as dependências da entrega.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\Executar-CitizenTech.ps1
```

Abra `http://127.0.0.1:8765`. A primeira execução baixa os pesos do modelo, o motor e o encoder. Esses artefatos somam cerca de 2,75 GB, além dos pacotes Python instalados pelo pip. Os pesos GGUF têm 2.497.281.120 bytes e o arquivo do motor tem 24.581.968 bytes. O inicializador verifica seus hashes SHA-256.

A aplicação usa Flask 3.0.2 e FastEmbed 0.7.3. O modelo e os embeddings rodam localmente em CPU. Os processos escutam em `127.0.0.1` nas portas 8080 e 8765. Use Ctrl+C para encerrar. Nas execuções seguintes, o inicializador reutiliza os artefatos de `.runtime`. O encoder procura primeiro os arquivos locais e baixa o que estiver faltando.

Para reproduzir a avaliação, encerre a aplicação e execute o comando abaixo no mesmo ambiente virtual.

```powershell
.\Executar-CitizenTech.ps1 -Avaliar
```

O comando executa os testes de software e a avaliação com o modelo real, depois atualiza `resultados.json`. Reinicie a aplicação e abra a aba **Avaliação** para consultar os casos e os julgamentos. Uma falha de comunicação interrompe a avaliação e preserva o último resultado completo.

## Arquitetura e configuração

A triagem em `triagem.py` preserva as regras de classificação do notebook de zeladoria da equipe. Categoria, prioridade, confiança heurística e indicação de revisão aparecem junto da orientação. A conversa guarda o trecho citado no turno anterior para resolver continuações e dá precedência a um novo tema explícito.

A recuperação combina BM25 e similaridade semântica por cosseno. O BM25 usa k1 = 1,2 e b = 0,75, normalização de acentos, palavras comuns e equivalências lexicais. O encoder é `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, executado em ONNX pelo FastEmbed, com mean pooling e vetores normalizados de 384 dimensões. A fusão RRF soma `1 / (60 + posição)` nos dois rankings e entrega até cinco candidatos em produção.

A avaliação de recuperação compara janela fixa com BM25, seção por assunto com BM25 e seção por assunto com busca híbrida, todas em top 3. A janela usa 32 palavras e até oito de sobreposição. A seção preserva o assunto e os metadados em um trecho, com limite de 100 palavras. IDs, URL, categoria e data de consulta acompanham as evidências.

Antes da seleção, `filtrar_servico` limita os candidatos quando a ação está explícita. Perguntas sobre varrição recebem seções desse serviço; relatos afirmativos de móveis, entulho ou restos de obra já abandonados em via pública recebem seções de remoção. Uma abertura de obra no asfalto não é tratada como resíduo descartado. Negação e coleta programada evitam forçar essa última rota. Se não houver fonte compatível, o sistema se abstém. Os IDs elegíveis ficam em `candidatos_modelo`, separados do ranking original em `recuperados`.

O filtro reconhece formulações explícitas por regras de palavras e ações. Relatos implícitos, negações complexas e perguntas sobre vários serviços continuam sujeitos às limitações do modelo. A referência por primeiro trecho recebe o mesmo filtro e escolhe o primeiro candidato elegível.

O gerador é Qwen3-4B-Instruct-2507, em GGUF Q4_K_M da Unsloth, executado por llama.cpp b12027. O modelo e sua distribuição GGUF indicam licença Apache 2.0. A execução usa contexto de 2.048 tokens, seis threads de CPU, um slot e zero camadas em GPU. Cada requisição usa temperatura 0, seed 42, até 48 tokens e timeout de 45 segundos. [Modelo original](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [pesos GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF), [llama.cpp b12027](https://github.com/ggml-org/llama.cpp/releases/tag/b12027).

O LLM recebe os candidatos e a pergunta e devolve um JSON contendo um ID permitido, `SEM_EVIDENCIA`, `FORA_ESCOPO` ou `ESCLARECER`. A aplicação valida a escolha, reproduz o texto do trecho e vincula a URL correspondente. A seleção extrativa facilita conferir a origem da orientação. Sua qualidade depende da recuperação e da adequação do trecho à pergunta.

O encoder utiliza o repositório ONNX `qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`. O snapshot usado nesta entrega é `faf4aa4225822f3bc6376869cb1164e8e3feedd0`, com o arquivo `model_optimized.onnx`, mantido em `.runtime/embeddings`. [Encoder ONNX](https://huggingface.co/qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q), [FastEmbed 0.7.3](https://github.com/qdrant/fastembed/releases/tag/v0.7.3).

## Conversa de exemplo

1. Pergunte **Como pedir reparo de uma lâmpada apagada?** e abra as evidências da resposta.
2. Na mesma conversa, envie **Como vejo o andamento?** para consultar a continuidade.
3. Pergunte **Uma passagem pública nunca teve iluminação. Qual serviço se aplica?** para comparar implantação e manutenção.
4. Envie **Qual é o horário exato que o Cata-Bagulho passa amanhã na minha rua?** para observar a abstenção diante de um dado local ausente da base.

Protocolos reais são abertos e acompanhados nos canais da Prefeitura. Use perguntas fictícias sem dados pessoais. As mensagens ficam na memória da aba e podem ser descartadas em **Nova conversa**.

## Avaliação e históricos

A avaliação guarda pergunta, evidências, resposta, status e julgamento. Compara o LLM com o baseline `primeiro_trecho`, que escolhe o candidato no topo usando a mesma busca e as mesmas proteções. Cada método mantém seu próprio estado de conversa. As métricas de sustentação consideram as orientações emitidas, e cada conjunto conserva seu denominador.

| Conjunto | Uso nesta entrega |
|---|---|
| `avaliacao_desenvolvimento.json` | 12 casos usados no desenvolvimento e nas escolhas de configuração. |
| `avaliacao_reservada.json` | 21 casos inicialmente reservados e depois usados como validação de diagnóstico. |
| `avaliacao_teste_final.json` | 20 casos da segunda validação, usados como diagnóstico após revelar novas falhas. |
| `avaliacao_confirmacao.json` | 16 casos construídos independentemente, sem acesso ao código ou aos resultados. A primeira execução foi reservada e depois passou a diagnóstico para os ajustes seguintes. |

Os conjuntos que orientaram correções são identificados como diagnóstico. A primeira execução reservada da confirmação acertou **13 de 16 casos** e motivou novos ajustes. A rodada posterior é identificada como **Reaplicação da confirmação**. `congelamento_confirmacao_inicial.json` preserva o congelamento anterior à primeira execução, e `congelamento_avaliacao.json` identifica a configuração reaplicada. As proporções descrevem amostras de conveniência vinculadas a este corpus.

| Arquivo | Conteúdo |
|---|---|
| `resultados_iniciais.json` | Histórico da configuração Qwen2.5-0.5B. |
| `resultados_validacao_1.json` | Primeira validação com Qwen2.5-3B, preservada antes dos ajustes decorrentes dela. |
| `resultados_teste_2.json` | Segunda validação com Qwen2.5-3B, preservada antes da busca híbrida e do modelo de 4B. |
| `resultados_confirmacao_inicial.json` | Primeira execução reservada da confirmação, preservada com seus 13/16 acertos e falhas antes dos ajustes seguintes. |
| `congelamento_confirmacao_inicial.json` | Hashes da configuração e da confirmação antes dessa primeira execução reservada. |
| `resultados.json` | Rodada publicada pelo avaliador, com configuração, comparação de componentes e quatro amostras reaplicadas após os ajustes. |

## Resultados publicados em 11/11/2026

A primeira confirmação reservada teve **13/16 acertos com LLM e 10/16 com o baseline**. A reaplicação anterior ao filtro de serviço teve **15/16 e 12/16**. Depois da correção, a confirmação reaplicada atendeu **16/16 e 12/16**, respectivamente. As reaplicações são medições de diagnóstico em perguntas já conhecidas.

| Conjunto na reaplicação | LLM | Primeiro trecho elegível |
|---|---|---|
| Desenvolvimento | 12/12 | 11/12 |
| Validação de diagnóstico | 21/21 | 16/21 |
| Validação de diagnóstico 2 | 20/20 | 16/20 |
| Reaplicação da confirmação | 16/16 | 12/16 |

O fluxo Q01-Q16 atendeu 16/16 critérios. Entre suas dez orientações emitidas, nenhuma foi reprovada por falta de suporte. A geração isolada acertou 10/11: em Q10, o modelo escolheu instruções de consulta para uma pergunta que exigia um horário. A proteção de dado local levou à abstenção no fluxo completo.

As duas falhas de seleção foram corrigidas: R10 agora usa DES.1, sobre remoção de resíduos abandonados em área pública; U05 usa DES.2, sobre varrição. O filtro de serviço limita os candidatos por ação explícita, e o LLM continua escolhendo entre as fontes elegíveis ou sinalizando abstenção. Abertura de obra no asfalto, entrega voluntária e coleta programada foram preservadas nos testes de regressão.

Os 85 registros do fluxo completo passaram nesta reaplicação. Os 22 testes de software também passaram. Esses resultados descrevem os casos conhecidos e a base pequena, sem garantir a mesma precisão em relatos novos ou implícitos. O arquivo `resultados_antes_filtro_servico.json` preserva as respostas incorretas anteriores e `congelamento_antes_filtro_servico.json` identifica sua configuração. Os contextos, candidatos elegíveis e respostas atuais estão em `resultados.json`.

## Arquivos e fontes

| Arquivo | Função |
|---|---|
| `app.py`, `templates/`, `static/` | Aplicação web e interface da conversa. |
| `assistente.py` | Segmentação, embeddings, busca híbrida, prompt, seleção e proteções. |
| `triagem.py` | Regras preservadas do notebook CitizenTech. |
| `corpus.json` | Trechos por assunto, URLs oficiais, categoria e data de consulta. |
| `municipios.json` | Snapshot com 5.298 nomes únicos de municípios da API oficial do IBGE, consultada em 10/11/2026. |
| `avaliar.py`, `test_assistente.py` | Avaliação dos componentes e do fluxo, comparação com baseline e testes de software. |
| `Relatorio_CitizenTech_Generative_AI.pdf` | Arquitetura, parâmetros, evidências, resultados e análise crítica da rodada final. |

As sete fontes de orientação pertencem à Prefeitura de São Paulo e têm URLs e datas em `corpus.json`. A lista auxiliar de municípios serve ao reconhecimento geográfico. [API de localidades do IBGE](https://servicodados.ibge.gov.br/api/docs/localidades), [endpoint de municípios](https://servicodados.ibge.gov.br/api/v1/localidades/municipios).

A base é estática. Pedidos de dados locais, inventário ou quantidades ausentes podem gerar abstenção. Filtros tratam padrões de dados pessoais e de injeção na entrada e na ingestão. As limitações e as falhas efetivamente observadas constam da avaliação e do relatório.

## Demonstração em vídeo

Por preferência da equipe, a entrega mantém o vídeo anterior com a narração já gravada. `Demonstracao_CitizenTech.mp4` tem 2min16,68s, em H.264, 1600×900, 25 quadros por segundo e áudio AAC estéreo. O arquivo foi restaurado integralmente do pacote anterior e passou pela decodificação de vídeo e áudio.

A gravação apresenta a configuração e as métricas da versão anterior do protótipo. O código e o relatório deste pacote permanecem corrigidos, com os resultados atuais e as correções de retirada de resíduos e varrição. Essa diferença está identificada também no relatório.

O áudio já está incluído. `Roteiro_CitizenTech_Ao_Vivo.md` contém um guia das cenas do vídeo mantido.
