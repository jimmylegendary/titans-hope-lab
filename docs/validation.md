# 실행 검증과 재현 범위

검증일은 2026-10-20이다. 공개할 monorepo 복사본에서 새 project-local 환경을 설치하고 실행했다. 환경은 Linux x86_64, Python 3.14.3, uv 0.10.2, PyTorch 2.9.1+cpu다. 두 프로젝트는 별도 `.venv`를 사용한다.

## CPU 검증

저장소 root에서 실행한다.

```bash
python scripts/verify_cpu.py --sync --report artifacts/cpu-validation.json
```

runner는 CPU lock으로 설치한 뒤 두 프로젝트를 순서대로 검사한다. 외부 dataset이나 pretrained 모델을 내려받지 않는다. 실제 실행의 전체 exit code는 0이었다.

| 프로젝트 | 검사 범위 | 통과 | Skip | 실패 | pytest 시간 |
|---|---|---:|---:|---:|---:|
| Titan-Llama | 전용 regression 7개와 선택한 Titans core CPU 검사 | 57 | 2 | 0 | 17.42s |
| HOPE | 현재 프로젝트의 전체 pytest suite | 12 | 1 | 0 | 13.27s |

Titan-Llama skip 2개는 현재 dense memory layout이 지원하지 않는 custom-memory 모델 검사다. HOPE skip 1개는 FineWeb-Edu shard가 있어야 실행하는 기존 200-step smoke다. Titan-Llama 전체 Cartesian test matrix와 GPU/flex-attention 검증은 실행하지 않았다.

원출력·선택한 test node·명령·설치 결과는 [cpu-validation.json](https://github.com/jimmylegendary/titans-hope-lab/blob/main/artifacts/cpu-validation.json)에 있다. 보고서의 machine-local 경로는 공개용 표기로 바꿨다. 이 표의 시간은 해당 검사 실행 시간이며 모델 성능 benchmark가 아니다.

검사는 freeze on/off, scratch 생성, optimizer update, cached decode, checkpoint 복원, matrix/deep memory gradient·상태 변화, aligned streaming, all-ignored partial tail과 gradient checkpointing을 포함한다. CLI integration은 실제 train/eval/generate/resume와 사용자 corpus 파일을 검사한다. checkpoint는 optimizer-step/새 stream 경계에서 재개하며 live memory state나 autograd graph를 저장하는 기능을 주장하지 않는다.

## 선택적 실제 pretrained 1B 검사

이 검사는 기본 CPU suite나 GitHub CPU workflow에 포함하지 않는다. 아래의 public model을 내려받고 사용할 수 있는 환경에서 선택해 실행한다.

```bash
implementations/titan-llama/.venv/bin/python scripts/verify_pretrained_titan_llama.py --download-only
implementations/titan-llama/.venv/bin/python scripts/verify_pretrained_titan_llama.py --local-only
```

model은 [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct), 고정 revision은 `5a8abab4a5d6f164389b1079fb721cfab8d7126c`다. 가중치는 repository에 포함하지 않는다. script는 `HF_HOME`을 존중하며 기본 cache는 `.cache/huggingface`다. 모델 실행 전 가용 메모리 14 GiB guard가 있다.

공개용 portable script를 기존 고정 revision cache와 함께 실행한 실제 결과는 **PASS, exit 0**이었다.

| 항목 | 관측 |
|---|---|
| Backbone | Llama 16 layers 중 `[0,1,2,3]` 교체 |
| 입력 | 16 tokens, segment 4, CPU FP32 |
| Backward | loss 7.337148189544678, NMM nonzero gradient tensor 44개 |
| 동결 | 원 backbone gradient 없음 |
| Update | AdamW 1 step 실행 |
| Cache | prefill 후 1-token decode, logits `(1,1,128256)` |
| 실행 자원 | 6.928s, peak RSS 6948.6 MiB |

수치는 [pretrained-1b-validation.json](https://github.com/jimmylegendary/titans-hope-lab/blob/main/artifacts/pretrained-1b-validation.json)의 실제 stage 출력이다. `source_git_head`는 가져온 pre-publication Titan snapshot을 뜻하며 public monorepo의 commit 번호를 뜻하지 않는다.

## 해석할 수 있는 범위

이 결과는 코드가 설치되고, 학습 신호와 memory update가 흐르며, 상태·저장·복원·추론 경로가 동작한다는 실행 증거다. 작은 corpus의 loss 감소나 몇 byte/token 생성으로 언어 유창성, 장문 recall, 지속 학습 품질 또는 논문의 accuracy·GPU latency 수치를 재현했다고 결론내리지 않는다.

가져온 runtime/test/config/license 48개 파일의 byte 동일성은 [source-integrity.json](https://github.com/jimmylegendary/titans-hope-lab/blob/main/artifacts/source-integrity.json), upstream과 local snapshot 기준은 [provenance.json](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/provenance.json)에 기록했다. GitHub CPU workflow는 [cpu.yml](https://github.com/jimmylegendary/titans-hope-lab/blob/main/.github/workflows/cpu.yml)에서 같은 runner를 실행한다.
