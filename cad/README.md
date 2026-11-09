# Draw CAD

CAD 2D próprio do Draw: um app de navegador em TypeScript, Canvas 2D e
sem dependências de runtime. Recebe ao vivo os traços confirmados no
papel (`python main.py --cad`) e também funciona sozinho.

## Rodar

```bash
npm ci
npm run build                 # gera cad/dist, servido pelo Python
cd .. && python main.py --cad # ou: python -m draw.cad_server --png canvas.png
```

Abre em `http://127.0.0.1:8770/`. Para desenvolver o front com recarga
automática, deixe o servidor Python rodando e use `npm run dev`. O Vite
encaminha `/api` para a porta 8770.

`npm run check` roda typecheck, build e os testes (`node --test`, que
executa os `.ts` direto; exige Node 22.18+).

## Uso

| Tecla | Ação |
|---|---|
| `V` `L` `P` `R` `C` `H` | Selecionar, Linha, Polilinha, Retângulo, Círculo, Mover vista |
| Roda / botão do meio / Espaço+arrastar | Zoom e mover vista |
| `F` | Enquadrar folha e desenho |
| `Shift` ao desenhar | Linha ortogonal |
| `Alt` | Desliga o snap no momento |
| `Enter` / duplo clique | Termina a polilinha (clicar no 1º ponto fecha) |
| `Esc` | Cancela ou termina e limpa a seleção |
| `Delete` | Apaga a seleção |
| Setas (`Shift` = 10 mm) | Move a seleção 1 mm |
| `Ctrl+Z` / `Ctrl+Shift+Z` / `Ctrl+Y` | Desfazer / refazer |
| `Ctrl+A` / `Ctrl+S` | Selecionar tudo / salvar JSON |

As coordenadas ficam em mm. O modelo usa o sistema do papel (origem no
topo, y para baixo). A barra de status, os campos de propriedade e o
DXF usam y para cima, como no CAD.

## Papel ao vivo

O servidor publica `GET /api/events` (SSE) com o estado da página:
`session`, `page`, `strokes` e `history`. Cada traço tem um `id`
estável, o tipo (`line`, `circle` ou `polyline`) e os pontos em mm.
Regras do cliente (`src/paper-sync.ts`):

- A camada **Papel (ao vivo)** espelha a página atual.
- Na virada de página, a página anterior é congelada em
  **Papel — página N**.
- Quando a sessão muda (o app foi reiniciado), o que estava ao vivo é
  congelado em **Papel (sessão anterior)**.
- Um traço do papel que você apagou, moveu ou editou fica desligado do
  papel e não volta na próxima atualização.

## Arquivos

- **Salvar/Abrir**: JSON `format: "draw-cad"`, `version: 1`.
- **DXF**: R12 (AC1009) em mm, com camadas e cores ACI. Gera `LINE`,
  `CIRCLE` e `POLYLINE`. Camadas ocultas não são exportadas.
- **SVG**: em mm, um `<g>` por camada visível.
