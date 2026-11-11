"""Regras de triagem do notebook Global_Solution_CitizenTech_RPA_IA.ipynb."""

import re
import unicodedata

PALAVRAS_POR_CATEGORIA = {
    'Iluminação pública': ['iluminacao', 'poste', 'postes', 'lampada', 'lampadas', 'luz apagada'],
    'Buraco ou pavimento': ['buraco', 'buracos', 'cratera', 'crateras', 'pavimento danificado'],
    'Resíduos e limpeza': ['lixo', 'entulho', 'residuos', 'coleta'],
    'Acessibilidade': ['rampa', 'rampas', 'calcada', 'calcadas', 'acessibilidade', 'piso tatil'],
}

CATEGORIAS_PRIORIDADE_MEDIA = ('Iluminação pública', 'Buraco ou pavimento', 'Resíduos e limpeza')

PADRAO_RISCO = r'\b(?:fio[s]? caido[s]?|fio[s]? exposto[s]?|fiacao exposta|risco de choque|faisca[s]?|alagamento[s]?|queda iminente)\b'

NEGACAO_EXPLICITA = r'\b(?:sem|nao ha) (?:risco de choque|faisca[s]?|alagamento[s]?)\b'

def normalizar(texto: str) -> str:
    texto = unicodedata.normalize('NFKD', texto.lower())
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r'[^\w\s]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def classificar_solicitacao(descricao: str) -> dict:
    texto = normalizar(descricao)
    pontuacoes = {
        categoria: sum(bool(re.search(rf'\b{re.escape(termo)}\b', texto)) for termo in termos)
        for categoria, termos in PALAVRAS_POR_CATEGORIA.items()
    }
    maior_pontuacao = max(pontuacoes.values())
    melhores = [categoria for categoria, pontos in pontuacoes.items() if pontos == maior_pontuacao]
    categoria = melhores[0] if maior_pontuacao and len(melhores) == 1 else 'Outros'
    confianca = min(maior_pontuacao / 2, 1.0) if categoria != 'Outros' else 0.0
    revisao = confianca < 1.0

    # shortcut: cobre negações literais; ampliar após avaliar relatos com outras formas de negação.
    texto_risco = re.sub(NEGACAO_EXPLICITA, '', texto)
    if re.search(PADRAO_RISCO, texto_risco):
        prioridade, revisao = 'Alta', True
    elif any(pontuacoes[categoria] for categoria in CATEGORIAS_PRIORIDADE_MEDIA):
        prioridade = 'Média'
    else:
        prioridade = 'Baixa'
    return {'category': categoria, 'priority': prioridade,
            'confidence': confianca, 'review_required': revisao}
