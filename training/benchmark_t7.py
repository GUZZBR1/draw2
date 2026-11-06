"""Bounded throughput and profiler sweep for the T7 segmentation workload."""

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _log(*parts):
    print(*parts, flush=True)


def _csv_ints(value, name, minimum, maximum_count=3):
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} precisa ser uma lista de inteiros") from exc
    if not values or len(values) > maximum_count:
        raise argparse.ArgumentTypeError(
            f"{name} aceita de 1 a {maximum_count} valores")
    if any(value < minimum for value in values):
        raise argparse.ArgumentTypeError(f"{name} precisa ser >= {minimum}")
    return list(dict.fromkeys(values))


def _sync(torch, device):
    if device.type == "tpu":
        torch.accelerator.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_stats(torch, device):
    if device.type == "cuda":
        return {
            "device_memory_allocated_bytes": int(
                torch.cuda.memory_allocated(device)),
            "device_memory_peak_bytes": int(
                torch.cuda.max_memory_allocated(device)),
        }
    if device.type == "tpu":
        stats = {}
        namespace = getattr(torch, "tpu", None)
        for result_key, candidates in (
                ("device_memory_allocated_bytes",
                 ("memory_allocated", "memory_allocated_bytes")),
                ("device_memory_peak_bytes",
                 ("max_memory_allocated", "max_memory_allocated_bytes"))):
            for name in candidates:
                fn = getattr(namespace, name, None)
                if callable(fn):
                    try:
                        stats[result_key] = int(fn())
                        break
                    except (TypeError, RuntimeError, NotImplementedError):
                        continue
        return stats
    return {}


def _profile(torch, train_step, loader, device, trace_dir, steps):
    activities = [torch.profiler.ProfilerActivity.CPU]
    device_activity = {"tpu": "TPU", "cuda": "CUDA"}[device.type]
    activity = getattr(torch.profiler.ProfilerActivity, device_activity, None)
    if activity is None:
        raise RuntimeError(f"torch.profiler não expõe ProfilerActivity.{device_activity}")
    activities.append(activity)

    profile_kwargs = {
        "activities": activities,
        "schedule": torch.profiler.schedule(
            wait=0, warmup=1, active=steps, repeat=1),
        "on_trace_ready": torch.profiler.tensorboard_trace_handler(
            str(trace_dir)),
    }
    if device.type == "tpu":
        try:
            from torch.tpu.profiler import TpuProfilerConfig
            profile_kwargs["experimental_config"] = TpuProfilerConfig(
                host_tracer_level=2, device_tracer_level=1,
                run_dir=trace_dir)
        except (ImportError, AttributeError, TypeError):
            _log("PROFILE_NOTE TpuProfilerConfig indisponível; usando config padrão")

    iterator = iter(loader)
    with torch.profiler.profile(**profile_kwargs) as profiler:
        for _ in range(steps + 1):
            images, masks = next(iterator)
            train_step(images, masks)
            _sync(torch, device)
            profiler.step()
    summary_path = trace_dir / "operator_summary.txt"
    try:
        summary_path.write_text(
            profiler.key_averages().table(row_limit=30), encoding="utf-8")
    except Exception as exc:  # profiler summary is supplemental to trace
        _log("PROFILE_SUMMARY_WARN", repr(exc))
    return str(trace_dir)


def _run_trial(torch, T, model, source_state, cfg, device, batch_size,
               num_workers, precision, warmup_steps, measured_steps,
               compile_model, data_dir):
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    model.load_state_dict(source_state)
    model.train()
    train_model = model
    if compile_model:
        backend = "tpu" if device.type == "tpu" else "inductor"
        train_model = torch.compile(model, backend=backend)

    training_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.4e-4,
        weight_decay=training_cfg["weight_decay"])
    class_weights = torch.tensor(
        cfg["loss"]["cross_entropy_class_weights"],
        dtype=torch.float32, device=device)
    dataset = T.build_dataset(str(data_dir), "train", cfg, augment=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=device.type == "cuda")
    iterator = iter(loader)
    use_amp = precision == "bf16" and device.type in ("tpu", "cuda")

    def train_step(images, masks):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16):
                logits = train_model(pixel_values=images).logits
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[-2:], mode="bilinear",
                        align_corners=False)
                loss = T.combined_loss(logits, masks, class_weights)
        else:
            logits = train_model(pixel_values=images).logits
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear",
                    align_corners=False)
            loss = T.combined_loss(logits, masks, class_weights)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    first_images, first_masks = next(iterator)
    train_step(first_images, first_masks)
    _sync(torch, device)
    first_step_s = time.perf_counter() - start

    for _ in range(max(0, warmup_steps - 1)):
        images, masks = next(iterator)
        train_step(images, masks)
    _sync(torch, device)

    data_wait_s = 0.0
    start = time.perf_counter()
    for _ in range(measured_steps):
        fetch_start = time.perf_counter()
        images, masks = next(iterator)
        data_wait_s += time.perf_counter() - fetch_start
        train_step(images, masks)
    _sync(torch, device)
    elapsed_s = time.perf_counter() - start
    stats = _memory_stats(torch, device)

    result = {
        "batch": batch_size,
        "num_workers": num_workers,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "compile": compile_model,
        "first_step_s": round(first_step_s, 3),
        "elapsed_s": round(elapsed_s, 3),
        "seconds_per_step": round(elapsed_s / measured_steps, 4),
        "samples_per_second": round(
            batch_size * measured_steps / elapsed_s, 2),
        "loader_wait_ms_per_step": round(
            data_wait_s * 1000 / measured_steps, 2),
        **stats,
    }
    del iterator, loader, optimizer, dataset
    return result, train_step


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark limitado e profiling do treino T7.")
    parser.add_argument("--device", choices=("tpu", "cuda"), default="tpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--batches", default="32,64")
    parser.add_argument("--workers", default="0,4,8")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=6)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)

    try:
        batches = _csv_ints(args.batches, "--batches", 1)
        workers = _csv_ints(args.workers, "--workers", 0)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if len(batches) * len(workers) > 6:
        parser.error("a busca aceita no máximo 6 combinações batch/workers")
    if args.compile and (len(batches) != 1 or len(workers) != 1):
        parser.error("--compile requer exatamente um batch e um valor de workers")
    if not (1 <= args.steps <= 20 and 1 <= args.warmup_steps <= 5
            and 2 <= args.profile_steps <= 10):
        parser.error("limites: steps 1-20, warmup-steps 1-5, profile-steps 2-10")

    import training.train_torchtpu as train_t7

    out_dir = Path(args.out_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = out_dir / "benchmarks" / stamp
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    train_t7._arm_compile_cache(run_dir)
    if args.device == "tpu":
        # Match train_torchtpu.py: libtpu reads scoped VMEM flags only
        # during initialization, before importing torch.
        train_t7._arm_scoped_vmem(train_t7.SCOPED_VMEM_DEFAULT_KIB)

    # TorchTPU reads its cache environment during torch import.
    import torch
    import training.train as T

    if args.device == "tpu":
        import torch_tpu  # noqa: F401
        if not torch.accelerator.is_available():
            raise SystemExit("TPU indisponível neste runtime")
        device = torch.device("tpu")
        props = torch.tpu.get_device_properties()
        hbm = (getattr(props, "hbm_bytes_per_device", None)
               or getattr(props, "hbm_bytes_per_chip", None))
        if args.precision == "bf16":
            norm_count = None
        else:
            norm_count = 0
    else:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA indisponível neste runtime")
        device = torch.device("cuda")
        hbm = torch.cuda.get_device_properties(device).total_memory
        if args.precision == "bf16" and not T.cuda_bf16_supported(torch, device):
            raise SystemExit("GPU sem bf16; rode com --precision fp32")
        norm_count = 0

    cfg = T.load_config(str(REPO_ROOT / "training" / "config.json"))
    data_dir = Path(args.data_dir)
    train_t7._ensure_data(
        data_dir, seed=42, n_train=args.train_count,
        n_val=args.val_count, n_test=args.test_count)
    torch.manual_seed(42)
    model_start = time.perf_counter()
    model = train_t7._build_model_t7(cfg, decoder=128)
    if device.type == "tpu" and args.precision == "bf16":
        norm_count = train_t7._patch_tpu_bf16_layer_norms(model)
    model = model.to(device)
    model_load_s = time.perf_counter() - model_start
    source_state = {key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
    _log("BENCH_DEVICE", json.dumps({
        "device": str(device), "name": getattr(props, "name", "cuda")
        if device.type == "tpu" else torch.cuda.get_device_name(device),
        "precision": args.precision, "hbm_bytes": hbm,
        "layer_norm_fp32_patch_count": norm_count,
        "model_load_s": round(model_load_s, 2),
        "compile": args.compile,
    }))

    results = []
    profile_step = None
    for batch_size in batches:
        for num_workers in workers:
            _log(f"BENCH_TRIAL batch={batch_size} workers={num_workers} "
                 f"steps={args.steps} warmup={args.warmup_steps}")
            result, train_step = _run_trial(
                torch, T, model, source_state, cfg, device,
                batch_size, num_workers, args.precision,
                args.warmup_steps, args.steps, args.compile, data_dir)
            results.append(result)
            _log("BENCH_RESULT", json.dumps(result))
            if profile_step is None:
                profile_step = (train_step, batch_size, num_workers)

    train_step, profile_batch, profile_workers = profile_step
    profile_loader = T.build_dataset(
        str(data_dir), "train", cfg, augment=True)
    from torch.utils.data import DataLoader
    profile_loader = DataLoader(
        profile_loader, batch_size=profile_batch, shuffle=True,
        drop_last=True, num_workers=profile_workers,
        persistent_workers=profile_workers > 0,
        prefetch_factor=2 if profile_workers > 0 else None,
        pin_memory=device.type == "cuda")
    profile_status = "ok"
    try:
        _profile(torch, train_step, profile_loader, device, trace_dir,
                 args.profile_steps)
    except Exception as exc:
        profile_status = f"failed: {type(exc).__name__}: {exc}"
        _log("PROFILE_FAILED", profile_status)

    report = {
        "created_utc": stamp,
        "device": str(device),
        "device_name": (getattr(props, "name", "tpu")
                        if device.type == "tpu"
                        else torch.cuda.get_device_name(device)),
        "precision": args.precision,
        "model": "SegFormer T7 decoder=128",
        "compile": args.compile,
        "hbm_bytes": hbm,
        "profile_status": profile_status,
        "trace_dir": str(trace_dir),
        "profile_steps": args.profile_steps,
        "results": results,
    }
    report_path = run_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = max(results, key=lambda result: result["samples_per_second"])
    _log("BENCH_BEST", json.dumps({
        "batch": best["batch"],
        "num_workers": best["num_workers"],
        "samples_per_second": best["samples_per_second"],
        "seconds_per_step": best["seconds_per_step"],
    }))
    _log("BENCHMARK_REPORT", report_path)
    _log("BENCHMARK_JSON", json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
