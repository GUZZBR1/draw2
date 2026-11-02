"""Tinta por modelo aprendido página→limpa (U-Net, ONNX).

O modelo recebe a página retificada (BGR uint8, qualquer tamanho) e
devolve, por pixel, a probabilidade de ser tinta manuscrita — pauta,
impresso, mão, caneta e sombra ficam fora. É a formulação do reinício
de 2026-10-31: em vez de diferenciar duas fotos sujas (referência,
tecla C, realinhamento), o modelo classifica a página direto e o app
só acumula a união das máscaras. Régua do treino (Kaggle, kernel
guzzbr/draw-page-clean-train-v0): IoU 0,80 com limiar 0,7 sobre
48 páginas held-out sintéticas, contra 0,18 do limiar de luminância.

Inferência por mosaico de 256×256 sem sobreposição, o mesmo
protocolo da avaliação do treino: a página é cortada em blocos de
256, a borda incompleta é preenchida com zeros, e os logits voltam
para a grade da página antes da sigmoide. Os blocos são empilhados
no eixo de lote e vão numa chamada só de session.run quando o
arquivo declara lote dinâmico: medido na auditoria de código de
2026-10-31, 20 blocos em um run() custaram 305 ms contra 372 ms em
20 runs de um bloco, com saída idêntica. O arquivo com lote fixo
continua um bloco por chamada, e DRAW_PAGE_CLEAN_BATCH=1 força o
caminho de um bloco por chamada para medir no alvo. A alternativa óbvia —
rodar a página em meia resolução e ampliar a máscara — foi medida e
reprovada nesta VM: IoU 0,85 → 0,22, porque o modelo aprendeu traço
fino na resolução cheia e a redução apaga o traço antes da rede. O
mosaico custa ~1,36 s por página de 864×1200 nesta VM (mediana de 20
páginas, 4 threads) e 168 ms no Ryzen do Lucas com 8 threads; no app
ele roda numa thread trabalhadora própria (draw.app._PageCleanWorker),
fora do laço principal, e o canvas é união, então página já lida não
precisa ser relida.

A sessão é preguiçosa: onnxruntime importado no primeiro uso,
erro explícito sem a dependência ou sem o arquivo. As threads
intra-op têm env própria (DRAW_PAGE_CLEAN_THREADS), separada da
do detector de folha: a passada completa mediu 205 ms com 4
threads e 168 ms com 8 no Ryzen do Lucas (perfil em
draw-lab/page-clean-v0/PERF_BASELINE_RYZEN.md), então o teto do
padrão desta sessão é 8 e o detector segue no 4 provado dele.
O padrão é limitado ao número de CPUs do host: pedir 8 threads
intra-op num host de 4 vCPUs degrada a passada em vez de
acelerá-la (medido na auditoria de 2026-11-02 nesta VM, com o
modelo w8 oficial de lote fixo: 2.150 ms com 8 threads contra
721 ms com 4 e 481 ms com 2 — thread sobrando vira contenção no
pool do ONNX Runtime). No Ryzen do Lucas (16 threads lógicas) o
limite não morde: o padrão continua 8, o valor medido dele.
DRAW_ORT_THREADS continua valendo como reserva, para as
ferramentas de perfil varrerem as duas sessões juntas.
"""

import os
from pathlib import Path

import numpy as np

from .manifest import verify_model

_PATCH = 256
_ENV_MODEL_VAR = "DRAW_PAGE_CLEAN_MODEL"
_ENV_THREADS_VAR = "DRAW_PAGE_CLEAN_THREADS"
_ENV_THREADS_FALLBACK_VAR = "DRAW_ORT_THREADS"
# Blocos por chamada de session.run. Sem a variável, os blocos da
# página vão todos numa chamada (o arquivo com lote dinâmico aceita
# e é a passada mais barata); DRAW_PAGE_CLEAN_BATCH=1 volta ao
# caminho antigo de um bloco por chamada, para comparar no alvo.
_ENV_BATCH_VAR = "DRAW_PAGE_CLEAN_BATCH"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "page_clean_unet.onnx"
)
DEFAULT_THRESHOLD = 0.7
# Medido no Ryzen (passada completa 205 -> 168 ms); ver docstring.
DEFAULT_THREADS = 8


def _default_threads():
    """Padrão de threads intra-op: DEFAULT_THREADS, sem passar do host.

    Num host com menos CPUs que o teto, o excesso de threads
    intra-op custa passada em vez de ganhar (medição na docstring
    do módulo). Sem contagem de CPUs disponível, vale o teto.
    """
    cpus = os.cpu_count() or DEFAULT_THREADS
    return max(1, min(DEFAULT_THREADS, cpus))


def _page_clean_threads(explicit=None):
    """Threads intra-op desta sessão: argumento > env própria >
    DRAW_ORT_THREADS > padrão do host. Inválido volta ao padrão."""
    if explicit:
        return int(explicit)
    for var in (_ENV_THREADS_VAR, _ENV_THREADS_FALLBACK_VAR):
        raw = os.environ.get(var)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                value = 0
            if value > 0:
                return value
    return _default_threads()


def _load_session(model_path, threads=None):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime nao esta instalado; o modelo pagina->limpa "
            "precisa dele. Instale com: pip install onnxruntime"
        ) from error
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"modelo ONNX nao encontrado: {path}. Passe o caminho do "
            f"arquivo .onnx ou defina a variavel {_ENV_MODEL_VAR}."
        )
    verify_model(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = _page_clean_threads(threads)
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _output_patch_size(session):
    """Lado do bloco na saída do modelo, e se o lote é dinâmico.

    O modelo do treino devolve o mesmo 256 do bloco. Um export que
    devolvesse outra resolução era aceito até a primeira escrita
    (ValueError de broadcast dentro da thread de tinta, que morria
    calada); aqui o contrato é lido na carga e a divergência vira
    erro nomeado.
    """
    output = session.get_outputs()[0]
    shape = list(output.shape)
    if len(shape) != 4:
        raise RuntimeError(
            "a saída do modelo pagina->limpa precisa ser [N, 1, H, W]; "
            f"o arquivo declara {shape}"
        )
    height, width = shape[2], shape[3]
    if not (isinstance(height, int) and isinstance(width, int)
            and height > 0 and width > 0):
        raise RuntimeError(
            "a saída do modelo pagina->limpa precisa ter altura e "
            f"largura fixas; o arquivo declara {shape}"
        )
    if height != width:
        raise RuntimeError(
            "a saída do modelo pagina->limpa precisa ser quadrada; "
            f"o arquivo declara {height}x{width}"
        )
    if height != _PATCH:
        raise RuntimeError(
            f"a saída do modelo pagina->limpa precisa ser {_PATCH}x{_PATCH} "
            f"(o mesmo lado do bloco); o arquivo declara {height}x{width}"
        )
    input_shape = list(session.get_inputs()[0].shape)
    batch = input_shape[0] if input_shape else 1
    dynamic_batch = not isinstance(batch, int)
    return height, dynamic_batch


def _env_batch_size():
    """DRAW_PAGE_CLEAN_BATCH: ausente = todos os blocos de uma vez."""
    raw = os.environ.get(_ENV_BATCH_VAR)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _batch_size(requested, tiles, dynamic_batch):
    """Blocos por chamada de session.run.

    Lote dinâmico no arquivo permite mandar os blocos de uma vez,
    que é a passada mais barata (medido: 20 blocos de 256 em um
    run() custa 305-310 ms contra 372-500 ms em 20 runs de um
    bloco, na VM da auditoria de 2026-10-31). Um arquivo com lote
    fixo continua no caminho de um bloco por run.
    """
    if not dynamic_batch:
        return 1
    if requested is None or requested <= 0:
        return max(1, tiles)
    return max(1, min(int(requested), tiles))


class PageCleanInk:
    """Máscara de tinta da página retificada, por mosaico de 256.

    O mosaico é montado uma vez (lista de recortes com o
    preenchimento das bordas) e mandado ao session.run em lotes: o
    arquivo com lote dinâmico roda todos os blocos numa chamada, e
    o arquivo com lote fixo continua um bloco por chamada. A conta
    é a mesma nos dois casos — a diferença medida é o custo fixo de
    cada run(), que o lote não paga.
    """

    def __init__(self, model_path=None, threshold=DEFAULT_THRESHOLD,
                 threads=None, batch_size=None):
        self.model_path = str(
            model_path
            or os.environ.get(_ENV_MODEL_VAR)
            or DEFAULT_MODEL_PATH
        )
        self.threshold = float(threshold)
        self.threads = _page_clean_threads(threads)
        self.batch_size = (
            batch_size if batch_size is not None
            else _env_batch_size()
        )
        self._session = None
        self._input_name = None
        self._dynamic_batch = False

    def load(self):
        """Carrega a sessão ONNX. Retorna True; erro sobe explícito."""
        self._session = _load_session(self.model_path, self.threads)
        self._input_name = self._session.get_inputs()[0].name
        self._dynamic_batch = _output_patch_size(self._session)[1]
        return True

    def _tiles(self, warped_bgr):
        """Recortes do mosaico, com a borda incompleta preenchida."""
        h, w = warped_bgr.shape[:2]
        tiles = []
        for y in range(0, h, _PATCH):
            for x in range(0, w, _PATCH):
                tile = warped_bgr[y:y + _PATCH, x:x + _PATCH]
                pad_h = _PATCH - tile.shape[0]
                pad_w = _PATCH - tile.shape[1]
                if pad_h or pad_w:
                    tile = np.pad(tile, [(0, pad_h), (0, pad_w), (0, 0)])
                tiles.append((y, x, tile))
        return h, w, tiles

    def _run_batch(self, batch):
        tensor = (
            np.stack(batch).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        )
        return self._session.run(None, {self._input_name: tensor})[0]

    def ink_prob(self, warped_bgr):
        """Probabilidade de tinta por pixel (float32 0..1), no
        tamanho do warp de entrada. É a saída crua do mosaico, antes
        do limiar: quem acumula passadas (confirmação entre passadas
        no app) precisa dela, porque o binário limiarizado joga fora
        a diferença entre tinta fraca estável e ruído de uma passada.
        """
        if self._session is None:
            self.load()
        h, w, tiles = self._tiles(warped_bgr)
        logits = np.zeros((h, w), dtype=np.float32)
        step = _batch_size(self.batch_size, len(tiles), self._dynamic_batch)
        for start in range(0, len(tiles), step):
            group = tiles[start:start + step]
            out = self._run_batch([tile for _y, _x, tile in group])
            for index, (y, x, _tile) in enumerate(group):
                hh = min(_PATCH, h - y)
                ww = min(_PATCH, w - x)
                logits[y:y + hh, x:x + ww] = out[index, 0, :hh, :ww]
        return 1.0 / (1.0 + np.exp(-logits))

    def ink_mask(self, warped_bgr):
        """Máscara uint8 0/255 da tinta, no tamanho do warp de entrada."""
        prob = self.ink_prob(warped_bgr)
        return np.where(prob > self.threshold, 255, 0).astype(np.uint8)
