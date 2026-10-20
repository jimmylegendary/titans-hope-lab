# Titans & HOPE research lab

이 저장소는 서로 다른 두 연구 구현을 한 public repository 안에 보존합니다. [Titan-Llama](https://github.com/jimmylegendary/titans-hope-lab/tree/main/implementations/titan-llama)는 사전학습 Llama attention layer 일부를 segmented attention과 외부 Neural Memory Module로 바꾸는 retrofit 실험입니다. [HOPE](https://github.com/jimmylegendary/titans-hope-lab/tree/main/implementations/hope)는 Self-Modifying Titans와 Continuum Memory System을 가진 소규모 byte-level 언어모델입니다. 두 구현을 결합한 단일 모델은 아닙니다.

원 논문과 구현 선택은 [paper-notes.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/paper-notes.md)에, pinned upstream과 이 snapshot의 차이는 [upstream-comparison.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/upstream-comparison.md)에, 실제 함수 호출 흐름은 [code-flow.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/code-flow.md)에 정리했습니다. GitHub 후보 조사와 감사 범위는 [repository-survey.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/repository-survey.md), 실행 조건과 실제 측정은 [validation.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/validation.md)에 있습니다.

## CPU 검증

Linux x86_64에서 Python 3.14.3을 준비하고 저장소 root에서 실행합니다.

```bash
python -m pip install uv==0.10.2
python scripts/verify_cpu.py --sync --report artifacts/cpu-validation.json
```

Runner는 프로젝트별 `uv.lock`으로 환경을 맞춘 뒤 두 테스트 묶음을 순차 실행합니다. 모델 가중치나 데이터셋은 받지 않으며, Python 패키지 wheel은 기존 cache를 사용할 수 있습니다. Titan-Llama의 미지원 custom-memory 검사 2건과 FineWeb-Edu 데이터가 필요한 HOPE 검사 1건은 skip됩니다. GPU 경로는 이 CPU 묶음에서 검사하지 않습니다. 세부 범위는 [validation.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/validation.md)에 있습니다. 한 프로젝트만 실행하려면 `--project hope` 또는 `--project titan-llama`를 지정합니다.

HOPE의 내장 byte corpus CLI는 [README.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/hope/README.md)에, Titan-Llama의 synthetic CPU smoke 및 학습 경로는 [README.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/titan-llama/README.md)에 있습니다.

## 선택적 Llama 1B 검사

캐시에 고정된 Llama-3.2-1B-Instruct가 있고 CPU 메모리 여유가 충분한 경우 실제 pretrained backbone 경로를 따로 확인할 수 있습니다.

```bash
implementations/titan-llama/.venv/bin/python \
  scripts/verify_pretrained_titan_llama.py --local-only
```

검사 스크립트는 기본 CPU suite에는 포함되지 않습니다. `--local-only`는 Hugging Face cache만 사용하고, cache가 없을 때는 실패합니다. `--download-only`는 고정 revision의 필요한 model 파일만 명시적으로 받습니다. 실행은 loading, backbone freeze, NMM gradient, optimizer update, cached decode를 확인할 뿐 언어 품질 benchmark나 논문 결과 재현은 아닙니다.

## 논문과 출처

- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)는 online associative neural memory와 세 가지 architecture pattern을 소개합니다.
- [It's All Connected: Miras](https://arxiv.org/abs/2504.13173)는 memory structure, attentional-bias objective, retention, optimizer를 설계 축으로 봅니다.
- [Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695)는 Self-Modifying Titans, CMS, HOPE를 제안합니다.
- [Titan-Llama](https://arxiv.org/abs/2610.19210)는 Llama-3.2-1B-Instruct의 일부 attention layer를 NMM 기반 blockwise path로 retrofit하는 방법을 보고합니다.

이 repository는 논문 저자의 공식 release가 아닙니다. Titan-Llama는 논문이 링크한 [Sarah Pan의 Titans PyTorch fork](https://github.com/sarahpannn/titans-pytorch/tree/c60b4efaa3a3c6df4deebd57cbccf75947b87cad)를 기반으로 하며, Titans core는 [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch/tree/1d40c445fa1794fb28582c721786482b5b683e47)의 코드 계보를 따릅니다. HOPE는 [smallhours19/nested-learning-hope](https://github.com/smallhours19/nested-learning-hope/tree/5fdbe522a9b10279661530aea5bf8dd7f9b4c180)의 공개 구현을 기반으로 합니다. Snapshot ID와 상속 관계는 [provenance.json](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/provenance.json), byte-level source 확인은 [source-integrity.json](https://github.com/jimmylegendary/titans-hope-lab/blob/main/artifacts/source-integrity.json), 저작권 표시는 [THIRD_PARTY_NOTICES.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/THIRD_PARTY_NOTICES.md)에 있습니다.

검증된 코드 실행과 논문 품질 재현은 구분합니다. CPU smoke, tiny corpus의 loss 감소, checkpoint 복원은 재현 가능한 실행 계약을 확인하는 증거이지, 논문의 대규모 accuracy·long-context recall·GPU latency를 입증하지 않습니다.
