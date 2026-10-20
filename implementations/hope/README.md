# HOPE: PyTorch 기반 Nested Learning

*Nested Learning: The Illusion of Deep Learning Architectures* 논문에 소개된 HOPE model과 메커니즘의 독립적인 소규모 PyTorch 구현입니다. 공식 구현이 아니며 논문의 대규모 결과를 재현했다고 주장하지 않습니다.

## 논문과 기반 코드

- Behrouz, Razaviyayn, Zhong, Mirrokni, [Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695).
- 기반 코드 참조는 [smallhours19/nested-learning-hope의 9ebfe97 commit](https://github.com/smallhours19/nested-learning-hope/tree/9ebfe97a186304fd06c692ecf0f5b96ab4a6cad9)에 고정했습니다.

소형 model은 [model.py](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/hope/hope/model.py)에, checkpoint와 CLI 동작은 [checkpoint.py](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/hope/hope/checkpoint.py), [cli.py](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/hope/hope/cli.py)에 있습니다. 논문에서 구현 세부를 정하지 않은 선택은 [DEVIATIONS.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/hope/docs/DEVIATIONS.md)에, 수치 안정성과 data 주의점은 [STABILITY.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/hope/docs/STABILITY.md)에 정리했습니다.

## 고정된 CPU 환경

Linux x86_64에서 이 디렉터리로 이동해 실행합니다. lock은 공식 PyTorch CPU index를 사용합니다.

```bash
uv sync --locked --extra train --extra dev
uv run python -m pytest -q
```

저장소 root의 순차 CPU runner:

```bash
python scripts/verify_cpu.py --project hope
```

HOPE 전체 테스트 suite를 CPU thread 1개로 실행하며 data나 model을 다운로드하지 않습니다.

## 소규모 학습, 평가, 생성, 재개

`hope-tiny` CLI는 작은 기본 model과 내장 byte corpus를 사용하므로 로컬 CPU smoke에 적합합니다.

```bash
uv run hope-tiny train --steps 4 --out runs/hope-tiny.pt
uv run hope-tiny eval --checkpoint runs/hope-tiny.pt
uv run hope-tiny generate --checkpoint runs/hope-tiny.pt --prompt "memory" --max-bytes 32
uv run hope-tiny train --resume runs/hope-tiny.pt --steps 4 --out runs/hope-tiny-resumed.pt
```

resume 명령은 corpus, memory mode, batch size, learning rate, seed가 checkpoint와 일치하는지 확인합니다. `--corpus PATH`로 로컬 byte-text 파일을 지정할 수 있으며, 기본값은 내장 corpus입니다.

더 큰 byte-training script에는 로컬 corpus 디렉터리가 필요합니다.

```bash
uv run python scripts/train_byte.py \
  --data data/fineweb-edu \
  --out runs/hope-byte \
  --max-steps 2
```

data는 저장소에 포함되지 않습니다. corpus 다운로드/준비는 선택 사항이며 위의 내장 CPU smoke 경로와 별개입니다. legacy data·figure 유틸리티도 외부 data/assets가 있어야 합니다.

## 범위와 license

이 snapshot은 실행 가능한 소형 architecture와 구현 선택을 다룹니다. 내장 CLI와 test suite는 논문 규모의 성능을 입증하지 않습니다. 기반 저장소의 [LICENSE](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/hope/LICENSE)를 보존했습니다.
