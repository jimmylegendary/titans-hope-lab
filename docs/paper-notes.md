# 네 편의 논문을 함께 읽는 메모

이 문서는 논문이 제안한 관점과 공개 코드에서 확인한 구현을 분리해 요약합니다. 논문 수치·주장은 각 원문에 귀속되며, 이 저장소의 실행 결과를 논문 성능 재현으로 해석하지 않습니다.

## Titans: test-time associative memory

[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)은 제한된 attention을 현재 문맥을 정밀하게 처리하는 단기 기억으로, neural memory를 과거 문맥을 압축하는 장기 기억으로 본다. Neural Memory Module은 key–value 연상을 내부 목적함수로 학습하고, test time에도 fast weights를 바꾼다. 저자들은 입력이 기존 기억에서 예상되지 않을 때 gradient 기반 surprise가 더 큰 update를 만들고, momentum과 decay가 과거 update를 누적·보존한다고 설명한다. Persistent memory는 현재 stream에 쓰는 context memory와 다른 task-level 파라미터다. 논문은 기억을 attention context에 더하는 MAC, gated memory MAG, attention과 memory를 층처럼 연결하는 MAL을 제시한다. 이 구분은 이 저장소의 Titan-Llama adapter를 이해하는 배경이며, 모든 Titans 변형이 한 구현에 들어 있다는 뜻은 아니다.

## MIRAS: memory 설계 선택지를 분해하기

[It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://arxiv.org/abs/2504.13173)은 서로 다른 sequence model을 associative memory로 해석한다. Miras는 설계 선택을 네 축으로 나눈다. memory 구조는 vector부터 MLP까지 달라질 수 있고, attentional bias는 key–value를 연결하는 내부 목적함수, retention은 새 기록과 이전 state의 균형, memory algorithm은 목적함수를 실제로 최적화하는 규칙을 정한다. 이 틀은 dot-product attention, L2 regression, 여러 retention gate, gradient update를 하나의 design space에서 비교하게 한다. Moneta, Yaad, Memora는 이 선택의 조합을 보여주는 별도 sequence model이다. Miras는 HOPE의 CMS나 Titan-Llama의 다른 이름이 아니며, 이 저장소의 런타임에 해당 세 모델을 구현했다고 주장하지 않는다.

## Nested Learning, CMS, Self-Modifying Titans, HOPE

[Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695)은 모델과 optimizer를 서로 다른 context flow와 update frequency를 가진 nested optimization problem으로 본다. CMS는 여러 MLP level을 연결하고, 각 level의 파라미터에 task objective의 gradient를 누적해 정해진 주기에 적용한다. 언어 모델링의 teaching signal은 다음 token 예측이다. Self-Modifying Titans는 key, value, query, learning-rate, retention 및 main memory가 자체 memory/update rule을 통해 context 중 변하는 구조다. HOPE는 이 fast self-modifying module 뒤에 상대적으로 느린 CMS를 둔다. M3는 CMS 아이디어를 momentum optimizer에 적용한 변형이며 DMGD는 gradient history를 압축하는 학습 가능한 deep memory optimizer다. 이들은 서로 대체 가능한 이름이나 여러 EMA의 다른 표기가 아니다.

## Titan-Llama: 사전학습 모델로의 retrofit

[Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules](https://arxiv.org/abs/2610.19210)은 Llama-3.2-1B-Instruct의 일부 attention layer를 blockwise attention과 외부 NMM으로 바꾸는 post-training 방법을 제안한다. 최근 segment는 local full-fidelity attention으로 남기고, 과거 key–value mapping은 고정 크기 MLP memory에 쓰며, query-dependent gate로 역사적 memory read와 local attention을 섞는다. NMM은 chunk별 self-supervised KV reconstruction 목적함수로 test time에도 업데이트된다. 논문은 원 backbone을 동결하고 작은 추가 모듈을 post-train하는 접근을 보고한다. 주요 정확도 설정은 첫 네 layer만 바꾸고 나머지 full-attention layer는 유지한다. 따라서 segmented-attention baseline과의 개선을 모든 layer 교체 결과나 full-attention 초과로 일반화하지 않는다. 이 공개 구현은 HOPE의 다중 CMS 계층이나 전체 self-modifying optimizer를 대신하지 않는 별도 retrofit 경로다.
