"""Testes da correção E1001 (VMEM scoped) do corpo TorchTPU e do
lançador run_t7: parsing/merging da flag
--xla_tpu_scoped_vmem_limit_kib no LIBTPU_INIT_ARGS e a detecção
do erro que dispara a escada de retry.

Só stdlib: os dois módulos são carregados pelo caminho e nenhum
deles importa torch no nível do módulo.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


body = _load("train_torchtpu_vmem", "training/train_torchtpu.py")
launcher = _load("run_t7_vmem", "training/run_t7.py")

FLAG = "--xla_tpu_scoped_vmem_limit_kib"


def test_scoped_vmem_in_parses_value():
    assert body._scoped_vmem_in(f"--a=1 {FLAG}=65536 --b") == 65536
    assert body._scoped_vmem_in("--a=1") is None
    assert body._scoped_vmem_in("") is None
    assert body._scoped_vmem_in(f"{FLAG}=xyz") is None


def test_with_scoped_vmem_appends_and_replaces():
    assert body._with_scoped_vmem("", 65536) == f"{FLAG}=65536"
    out = body._with_scoped_vmem(f"--a=1 {FLAG}=16384", 131072)
    assert f"{FLAG}=131072" in out
    assert "16384" not in out
    assert "--a=1" in out
    assert body._with_scoped_vmem("--a=1", 0) == "--a=1"


def test_arm_scoped_vmem_sets_default(monkeypatch):
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    assert body._arm_scoped_vmem(65536) == 65536
    import os
    assert f"{FLAG}=65536" in os.environ["LIBTPU_INIT_ARGS"]


def test_arm_scoped_vmem_respects_existing_flag(monkeypatch):
    monkeypatch.setenv("LIBTPU_INIT_ARGS", f"--a=1 {FLAG}=999")
    assert body._arm_scoped_vmem(65536) == 999
    import os
    assert os.environ["LIBTPU_INIT_ARGS"] == f"--a=1 {FLAG}=999"


def test_arm_scoped_vmem_disabled_and_clamped(monkeypatch):
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    assert body._arm_scoped_vmem(0) is None
    import os
    assert FLAG not in os.environ.get("LIBTPU_INIT_ARGS", "")
    assert body._arm_scoped_vmem(10**9) == body.SCOPED_VMEM_MAX_KIB


def test_is_scoped_vmem_oom_body():
    assert body._is_scoped_vmem_oom(
        Exception("e1001: CompileTimeScopedVmemOom: Ran out"))
    assert body._is_scoped_vmem_oom(
        Exception("exceeded scoped vmem limit by 8.01M"))
    assert not body._is_scoped_vmem_oom(Exception("outro erro"))


def test_launcher_env_with_scoped_vmem(monkeypatch):
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--a=1")
    env = launcher._env_with_scoped_vmem(131072)
    assert f"{FLAG}=131072" in env["LIBTPU_INIT_ARGS"]
    assert "--a=1" in env["LIBTPU_INIT_ARGS"]
    monkeypatch.setenv("LIBTPU_INIT_ARGS", f"{FLAG}=65536")
    env = launcher._env_with_scoped_vmem(131072)
    assert f"{FLAG}=131072" in env["LIBTPU_INIT_ARGS"]
    assert "65536" not in env["LIBTPU_INIT_ARGS"]


def test_launcher_detects_e1001():
    assert launcher._is_scoped_vmem_oom(
        "BackendCompilerFailed ... CompileTimeScopedVmemOom")
    assert launcher._is_scoped_vmem_oom(
        "ERRO_E1001: CompileTimeScopedVmemOom — ...")
    assert not launcher._is_scoped_vmem_oom("DELIVERY validated")
