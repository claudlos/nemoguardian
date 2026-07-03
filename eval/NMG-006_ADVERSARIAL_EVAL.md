# NMG-006 — Adversarial eval of the moderation cascade

**Goal:** quantify how often adversarial text bypasses the cascade (verdict
`safe` on a harmful probe). This is the one open High finding in SECURITY.md;
publishing a measured bypass rate closes the loop.

Everything here is prepared; the run itself needs the real models on a GPU host.

## How it works

garak's `rest.RestGenerator` fires its attack probes at `/demo/moderate` and
records the cascade's `verdict` field as the "model output"
(`eval/garak_rest_config.json`). Since every probe prompt is adversarial by
construction, `verdict == "safe"` = bypass. `eval/analyze_bypass.py` turns the
garak report into per-probe and overall bypass rates.

## Run it (on the GPU host)

```bash
# 1. Serve the real cascade with the demo endpoint enabled (local only!)
NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=1 NEMOGUARDIAN_RATE_LIMIT_RPM=0 \
    uvicorn nemoguardian.server:app --port 8000 &

# 2. Point garak at it (pip install garak)
python -m garak \
    --model_type rest.RestGenerator \
    --generator_option_file eval/garak_rest_config.json \
    --probes promptinject,dan.Dan_11_0,encoding,goodside \
    --generations 1 \
    --report_prefix eval/results/nmg006

# 3. Bypass rate
python eval/analyze_bypass.py eval/results/nmg006.report.jsonl
```

Notes:
- `NEMOGUARDIAN_RATE_LIMIT_RPM=0` — garak will exceed the default 60 rpm.
- Start with the probe set above (~10-20 min on an RTX 3090); a full
  `--probes all` sweep is the publishable number but takes hours.
- Try `mode: "deep"` in `garak_rest_config.json` (with `NVIDIA_API_KEY` set)
  to measure how much the triage stage lowers the bypass rate vs `standard` —
  that delta is the headline claim ("vendor diversity slows transfer") for the
  follow-up paper.
- Publish results in SECURITY.md next to NMG-006 and in the README benchmark
  section: overall bypass rate + the 3 worst probes.
