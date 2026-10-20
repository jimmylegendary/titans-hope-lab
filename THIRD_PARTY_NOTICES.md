# Third-party code and attribution

이 저장소는 공개 구현을 고정 commit에서 가져와 실행 오류 수정, 검증, 재현 진입점과 해설을 더한 연구용 모음이다. 원 논문과 upstream 구현의 저작자를 아래에 구분한다.

| 경로 | 직접 기반 | upstream commit | 원 라이선스 |
|---|---|---|---|
| `implementations/titan-llama/` | [sarahpannn/titans-pytorch](https://github.com/sarahpannn/titans-pytorch/tree/c60b4efaa3a3c6df4deebd57cbccf75947b87cad) | `c60b4efaa3a3c6df4deebd57cbccf75947b87cad` | MIT, Copyright (c) 2025 Phil Wang |
| `implementations/hope/` | [smallhours19/nested-learning-hope](https://github.com/smallhours19/nested-learning-hope/tree/5fdbe522a9b10279661530aea5bf8dd7f9b4c180) | `5fdbe522a9b10279661530aea5bf8dd7f9b4c180` | MIT, Copyright (c) 2026 the nested-learning-hope authors |

Titan-Llama의 직접 기반은 *Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules*가 연결하는 저자 저장소다. 이 fork는 [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch/tree/1d40c445fa1794fb28582c721786482b5b683e47)의 Titans core를 계승한다. Titan-Llama 논문과 retrofit 구현은 Sarah Pan·Ali Behrouz 및 해당 repository contributors에게, 원 Titans core는 Phil Wang 및 해당 contributors에게 귀속된다.

각 subtree의 원 MIT LICENSE를 그대로 보존했다: Titan-Llama [LICENSE](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/titan-llama/LICENSE), HOPE [LICENSE](https://github.com/jimmylegendary/titans-hope-lab/blob/main/implementations/hope/LICENSE). 본 저장소의 추가 문서와 glue code에도 root [LICENSE](https://github.com/jimmylegendary/titans-hope-lab/blob/main/LICENSE)의 MIT 조건을 적용한다. 원 저작권 고지를 대체하지 않는다.

논문 원문·그림·dataset·pretrained 모델 가중치는 이 저장소에서 재배포하지 않는다. 논문은 원문 링크로 인용하며, 데이터와 모델이 필요한 선택 실험은 원 제공자의 조건을 따른다. upstream의 대형 dataset과 참고 그림/결과 assets는 code snapshot에서 제외했다. 데이터 없이 실행하는 CPU 경로와 재현 명령을 별도로 제공한다.

출처와 snapshot 기준은 [provenance.json](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/provenance.json), 변경점은 [upstream-comparison.md](https://github.com/jimmylegendary/titans-hope-lab/blob/main/docs/upstream-comparison.md)에 기록한다.
