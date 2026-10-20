# Titan-Llama 연구 구현

Hugging Face Llama backbone에 segmented attention과 Titans Neural Memory Module(NMM)을 적용하는 독립 연구 구현입니다. 공식 배포본이 아니며 논문의 대규모 결과를 재현했다고 주장하지 않습니다.

## 논문과 기반 코드

- Behrouz 외, [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663).
- Pan, Behrouz, [Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules](https://arxiv.org/abs/2610.19210).
- 기반 코드 참조는 [sarahpannn/titans-pytorch의 c60b4ef commit](https://github.com/sarahpannn/titans-pytorch/tree/c60b4efaa3a3c6df4deebd57cbccf75947b87cad)에 고정했습니다. 이 저장소 snapshot에는 런타임과 검증 변경이 포함됩니다.

모델 wrapper는 [titan_llama.py](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/titan-llama/titan_llama.py)에, 학습과 checkpoint 처리는 [train_titan_llama.py](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/titan-llama/train_titan_llama.py)에 있습니다. `configs/`의 JSON 9개는 논문 실험 설정을 보존합니다. pretrained 모델의 architecture 차원은 Hugging Face config에서 읽으며, 이 JSON들은 CPU smoke 크기의 설정이 아닙니다.

## 고정된 CPU 환경과 smoke 검사

Linux x86_64에서 이 디렉터리로 이동해 실행합니다. lock은 공식 PyTorch CPU index의 CPU 전용 PyTorch를 설치합니다.

```bash
uv sync --locked --extra research --extra test
.venv/bin/python scripts/titan_llama_smoke.py --output-dir outputs/tiny-cpu
.venv/bin/python -m pytest -q tests/test_titan_llama_runtime.py
.venv/bin/python scripts/pretrain.py 1024 --dry-run
```

[titan_llama_smoke.py](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/titan-llama/scripts/titan_llama_smoke.py)는 random 초기화한 2-layer, hidden 64 Llama와 synthetic token ID를 씁니다. optimizer update 한 번, evaluation, cached generation, checkpoint 저장과 state restore를 실행합니다. 모델·dataset을 다운로드하거나 W&B run을 시작하지 않습니다. 제한된 CPU 검증 테스트는 [test_titan_llama_runtime.py](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/titan-llama/tests/test_titan_llama_runtime.py)에 있습니다.

저장소 root에서 CPU runner를 실행할 수 있습니다.

```bash
python scripts/verify_cpu.py --project titan-llama
```

## 논문 설정 학습

논문 preset은 긴 context, pretrained weight, streaming dataset을 사용하므로 CPU smoke 환경 대상이 아닙니다. 호스트 CUDA runtime에 맞는 PyTorch가 설치된 별도 GPU 환경을 사용하고 CPU lock은 그 환경에 sync하지 마세요. 이 fork는 GPU PyTorch를 유지하는 환경에서 `pip install -e '.[research,test]'`로 editable 설치할 수 있습니다.

pretraining preset은 기본으로 `unsloth/Llama-3.2-1B-Instruct`를 사용하고 distributed training을 켭니다. 이 디렉터리에서 `torchrun`으로 시작합니다.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/pretrain.py 1024 --output-dir outputs/seg1024_pretrain
```

같은 output 디렉터리의 `latest_checkpoint.pt`에서 재개합니다.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/pretrain.py 1024 --output-dir outputs/seg1024_pretrain --resume
```

학습은 선택한 Llama weight와 streaming dataset을 다운로드할 수 있으며 GPU memory, 저장 공간, 네트워크가 필요합니다. 실제 학습 없이 설정만 볼 때는 `scripts/pretrain.py 1024 --dry-run`을 실행합니다.

## 학습 checkpoint 평가와 생성

아래 코드는 trainer checkpoint를 불러 짧은 prompt의 loss를 계산하고 이어서 생성합니다. 학습 후 같은 GPU 환경에서 실행하며, cache가 없으면 base model과 tokenizer를 받을 수 있습니다.

```bash
python - <<'PY'
import torch
from transformers import AutoTokenizer
from titan_llama import TitanLLaMAForCausalLM

base = "unsloth/Llama-3.2-1B-Instruct"
model = TitanLLaMAForCausalLM.from_pretrained(
    "outputs/seg1024_pretrain/latest_checkpoint.pt",
    base_model_name_or_path=base,
    dtype=torch.bfloat16,
    device="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(base)
ids = tokenizer("Explain how memory updates across chunks.", return_tensors="pt")["input_ids"].to("cuda")
model.eval()
with torch.no_grad():
    metrics = model(input_ids=ids, labels=ids, use_cache=False)
    generated = model.generate_with_titan_memory(
        input_ids=ids,
        max_new_tokens=32,
        do_sample=False,
    )
print("loss=", float(metrics["loss"]))
print(tokenizer.decode(generated[0], skip_special_tokens=True))
PY
```

The optional [verify_pretrained_titan_llama.py](https://github.com/jimmylegendary/titans-hope-lab/blob/main/scripts/verify_pretrained_titan_llama.py) checks the pinned 1B backbone on CPU with a 14 GiB available-memory guard. It is not part of CPU CI. Pass `--local-only` to require an existing Hugging Face cache, or `--download-only` to fetch only the pinned model files.

## 범위와 한계

CPU smoke는 작은 random 모델로 실행 경로만 확인하며 품질 benchmark가 아닙니다. CPU suite는 generic Titans 전체 parameter matrix, 전체 long-context sampling matrix, GPU 전용 flex/accelerated scan 테스트를 실행하지 않습니다. 실제 pretrained 1B 실행에는 큰 메모리가 필요하며 별도 선택 검사입니다.

기반 프로젝트의 [LICENSE](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/titan-llama/LICENSE)를 보존했습니다. 기존 데이터·figure 유틸리티는 외부 data/assets가 필요하며 CPU no-data 경로에는 포함되지 않습니다.
