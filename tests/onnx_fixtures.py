"""Modelos ONNX minúsculos para os testes de contrato, gerados na hora.

Os testes de `draw.neural` e `draw.page_clean` precisam de um arquivo
ONNX de verdade: o valor deles é justamente passar pelo ONNX Runtime
(ler nome da entrada, forma das saídas, lote dinâmico), o que uma
sessão falsa em Python não exercita. Gerar o arquivo no teste evita
versionar binário novo e mantém os modelos (e o que eles devolvem)
visíveis no código.

O pacote `onnx` só é necessário para *gerar* o arquivo. Sem ele, os
testes que dependem destes fixtures saem como skip — o onnxruntime
continua sendo o único requisito de execução.
"""

import numpy as np


def _build(nodes, inputs, outputs, initializers, path):
    import onnx
    from onnx import helper

    graph = helper.make_graph(nodes, "draw-test-model", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _initializer(name, shape, seed=0, scale=0.05):
    from onnx import numpy_helper

    rng = np.random.default_rng(seed)
    values = (rng.standard_normal(shape) * scale).astype(np.float32)
    return numpy_helper.from_array(values, name)


def sheet_model(path, input_name="image", size=64, channels=2):
    """Segmentador de folha: máscara de `channels` logits + presença.

    Com channels=2 o contrato é o legado (fundo, papel). Com
    channels=4 é o export híbrido (fundo, papel, mão, tinta) com
    cabeça de presença treinada.

    A máscara sai de uma convolução 1x1 sobre a entrada, então
    papel é a região com pixel claro; a presença é a média do canal
    vermelho escalada. O teste não julga a semântica: ele verifica
    o contrato e a leitura das saídas.
    """
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node(
            "Conv",
            [input_name, "w_mask", "b_mask"],
            ["mask_logits"],
            kernel_shape=[1, 1],
        ),
        helper.make_node("GlobalAveragePool", [input_name], ["pooled"]),
        helper.make_node(
            "ReduceMean", ["pooled"], ["mean_rgb"], axes=[1, 2, 3], keepdims=1
        ),
        helper.make_node("Mul", ["mean_rgb", "gain"], ["presence_raw"]),
        helper.make_node(
            "Reshape", ["presence_raw", "presence_shape"], ["presence_logit"]
        ),
    ]
    initializers = [
        _initializer("w_mask", (channels, 3, 1, 1), seed=1),
        _initializer("b_mask", (channels,), seed=2),
        helper.make_tensor("gain", TensorProto.FLOAT, [1], [20.0]),
        helper.make_tensor("presence_shape", TensorProto.INT64, [1], [1]),
    ]
    return _build(
        nodes,
        [
            helper.make_tensor_value_info(
                input_name, TensorProto.FLOAT, [1, 3, size, size]
            )
        ],
        [
            helper.make_tensor_value_info(
                "mask_logits", TensorProto.FLOAT, [1, channels, size, size]
            ),
            helper.make_tensor_value_info("presence_logit", TensorProto.FLOAT, [1]),
        ],
        initializers,
        path,
    )


def four_class_model(path, input_name="image", size=64, extra_output=False):
    """SegFormer de quatro classes exportado só como 'logits'."""
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("Conv", [input_name, "w", "b"], ["raw"], kernel_shape=[1, 1]),
        helper.make_node(
            "Resize", ["raw", "", "scales", ""], ["logits"], mode="nearest"
        ),
    ]
    initializers = [
        _initializer("w", (4, 3, 1, 1), seed=3),
        _initializer("b", (4,), seed=4),
        helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 1.0, 1.0]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4, size, size])
    ]
    if extra_output:
        nodes.append(helper.make_node("Identity", ["logits"], ["aux"]))
        outputs.append(
            helper.make_tensor_value_info("aux", TensorProto.FLOAT, [1, 4, size, size])
        )
    return _build(
        nodes,
        [
            helper.make_tensor_value_info(
                input_name, TensorProto.FLOAT, [1, 3, size, size]
            )
        ],
        outputs,
        initializers,
        path,
    )


def page_clean_model(path, patch=64, dynamic_batch=True, output_scale=1):
    """U-Net didático página->limpa: devolve um logit por pixel.

    O logit é proporcional à luminância do bloco (tinta escura =
    logit baixo), o que dá uma máscara não trivial para comparar o
    caminho em lote com o caminho de um bloco por chamada. Com
    output_scale != 1 a saída sai em outra resolução, para testar
    a validação do contrato.
    """
    from onnx import TensorProto, helper

    out = patch // output_scale
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[1], keepdims=1),
    ]
    if output_scale == 1:
        nodes.append(helper.make_node("Identity", ["mean"], ["logits"]))
        initializers = []
    else:
        nodes.append(
            helper.make_node(
                "Resize", ["mean", "", "scales", ""], ["logits"], mode="nearest"
            )
        )
        initializers = [
            helper.make_tensor(
                "scales",
                TensorProto.FLOAT,
                [4],
                [1.0, 1.0, 1.0 / output_scale, 1.0 / output_scale],
            )
        ]
    batch = "N" if dynamic_batch else 1
    return _build(
        nodes,
        [
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, [batch, 3, patch, patch]
            )
        ],
        [
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, [batch, 1, out, out]
            )
        ],
        initializers,
        path,
    )
