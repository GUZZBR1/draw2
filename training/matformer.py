"""MatFormer: bloco Transformer aninhado para inferencia elastica.

Implementacao do mecanismo de "MatFormer: Nested Transformer for
Elastic Inference" (Devvrit et al., arXiv:2310.07707, NeurIPS
2024), na receita da versao final do papel (v2, dez/2024):

- So a dimensao oculta da FFN e aninhada: o submodelo de
  granularidade i usa os PRIMEIROS m_i neuronios da FFN inteira.
  Atencao, embeddings e profundidade sao compartilhados (o
  aninhamento de cabecas do papel e so um experimento de
  apendice; a elasticidade de profundidade do Gemma 3n e da
  produtizacao, nao do papel).
- Razoes de granularidade {0.5, 1, 2, 4} sobre a largura-base da
  FFN (S/M/L/XL do papel).
- Treino: cada passo SORTEIA uma granularidade (uniforme) e
  treina so ela — os parametros compartilhados recebem gradiente
  de todas, e e por isso que, medido no papel, os submodelos
  menores empatam ou vencem modelos independentes do mesmo
  tamanho.
- Inferencia: qualquer granularidade sai por fatiamento dos
  pesos, sem retreino; larguras diferentes por camada
  (Mix'n'Match) tambem funcionam, porque cada bloco le a sua
  granularidade.

Papel no Draw (decisao de 2026-11-06): isto e o bloco de
construcao do experimento do modelo unico — UM artefato com
varios tamanhos de inferencia, para a app escolher pelo hardware
(a latencia em CPU sem GPU e um gargalo medido do produto). NAO
esta ligado a nenhuma configuracao de producao: os modelos de
producao sao CNNs U-Net, onde o aninhamento nao se aplica, e a
promocao de qualquer modelo segue a bancada + holdout + mesa.
"""

import torch
from torch import nn
from torch.nn import functional as F


# Razoes S/M/L/XL do papel sobre a largura-base da FFN.
GRANULARITY_RATIOS = (0.5, 1.0, 2.0, 4.0)


def granularity_widths(base_hidden, ratios=GRANULARITY_RATIOS):
    """Larguras ocultas (crescentes) das granularidades."""
    widths = []
    for ratio in ratios:
        width = int(round(base_hidden * ratio))
        if width < 1:
            raise ValueError(f"largura {width} invalida para base {base_hidden}")
        if widths and width <= widths[-1]:
            raise ValueError(
                f"larguras das granularidades devem ser crescentes: {widths + [width]}"
            )
        widths.append(width)
    return tuple(widths)


class NestedFFN(nn.Module):
    """FFN com os submodelos aninhados na dimensao oculta.

    Guarda os pesos da largura maior; a passada com
    granularidade g usa as primeiras widths[g] linhas de W1 e as
    primeiras widths[g] colunas de W2. O gradiente de um passo
    numa granularidade pequena nao toca a cauda — e essa a
    mecanica que o treino por sorteio explora.
    """

    def __init__(self, d_model, base_hidden, ratios=GRANULARITY_RATIOS):
        super().__init__()
        self.widths = granularity_widths(base_hidden, ratios)
        max_width = self.widths[-1]
        self.fc1 = nn.Linear(d_model, max_width)
        self.fc2 = nn.Linear(max_width, d_model)
        self.act = nn.GELU()

    def forward(self, x, granularity):
        width = self.widths[granularity]
        hidden = F.linear(x, self.fc1.weight[:width], self.fc1.bias[:width])
        hidden = self.act(hidden)
        return F.linear(hidden, self.fc2.weight[:, :width], self.fc2.bias)


class MatBlock(nn.Module):
    """Bloco pre-norma: atencao compartilhada + FFN aninhada."""

    def __init__(self, d_model, num_heads, base_hidden, ratios=GRANULARITY_RATIOS):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = NestedFFN(d_model, base_hidden, ratios)

    def forward(self, x, granularity):
        residual = self.norm1(x)
        attended, _ = self.attn(residual, residual, residual)
        x = x + attended
        return x + self.ffn(self.norm2(x), granularity)


class _PlainFFN(nn.Module):
    """FFN comum de largura fixa: o destino da extracao."""

    def __init__(self, d_model, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _PlainBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _PlainFFN(d_model, hidden)

    def forward(self, x):
        residual = self.norm1(x)
        attended, _ = self.attn(residual, residual, residual)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class PlainEncoder(nn.Module):
    """Encoder extraido: blocos comuns, sem aninhamento.

    E o artefato que a inferencia roda quando escolhe um
    tamanho: mesmos pesos do MatEncoder na granularidade
    extraida, nenhuma logica condicional.
    """

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class MatEncoder(nn.Module):
    """Pilha de blocos MatFormer com extracao de submodelos."""

    def __init__(
        self, d_model, num_heads, base_hidden, depth, ratios=GRANULARITY_RATIOS
    ):
        super().__init__()
        self.widths = granularity_widths(base_hidden, ratios)
        self.blocks = nn.ModuleList(
            MatBlock(d_model, num_heads, base_hidden, ratios) for _ in range(depth)
        )

    def forward(self, x, granularity):
        """granularity: um indice (todas as camadas) ou uma lista
        por camada (Mix'n'Match). A forma da saida nao depende da
        granularidade — so o custo muda."""
        if isinstance(granularity, int):
            granularities = [granularity] * len(self.blocks)
        else:
            granularities = list(granularity)
            if len(granularities) != len(self.blocks):
                raise ValueError(
                    "Mix'n'Match exige uma granularidade por camada: "
                    f"{len(granularities)} para {len(self.blocks)}"
                )
        for block, g in zip(self.blocks, granularities):
            x = block(x, g)
        return x

    def sample_granularity(self, generator=None):
        """Sorteio uniforme da receita v2: um indice por passo."""
        return int(torch.randint(len(self.widths), (1,), generator=generator).item())

    def parameter_count(self, granularity=None):
        """Parametros do modelo cheio (None) ou do extraido."""
        if granularity is None:
            return sum(p.numel() for p in self.parameters())
        return sum(p.numel() for p in self.extract(granularity).parameters())

    @torch.no_grad()
    def extract(self, granularity):
        """Submodelo autonomo da granularidade, por fatiamento.

        Propriedade definidora do MatFormer: a saida do extraido
        e IDENTICA a do modelo cheio rodando na mesma
        granularidade — os pesos sao os mesmos, so falta a cauda.
        """
        blocks = []
        for block in self.blocks:
            width = block.ffn.widths[granularity]
            plain = _PlainBlock(
                block.norm1.normalized_shape[0],
                block.attn.num_heads,
                width,
            )
            plain.norm1.load_state_dict(block.norm1.state_dict())
            plain.attn.load_state_dict(block.attn.state_dict())
            plain.norm2.load_state_dict(block.norm2.state_dict())
            plain.ffn.fc1.weight.copy_(block.ffn.fc1.weight[:width])
            plain.ffn.fc1.bias.copy_(block.ffn.fc1.bias[:width])
            plain.ffn.fc2.weight.copy_(block.ffn.fc2.weight[:, :width])
            plain.ffn.fc2.bias.copy_(block.ffn.fc2.bias)
            blocks.append(plain)
        encoder = PlainEncoder(blocks)
        encoder.eval()
        return encoder
