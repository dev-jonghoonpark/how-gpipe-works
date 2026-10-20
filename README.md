# How GPipe Works

**GPipe — 층을 나누고, 배치를 쪼개고, 갱신은 한 번에.** Huang et al.(NeurIPS 2019)의 「GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism」을 브라우저에서 직접 돌려 보는 인터랙티브 교육 자료입니다. 모델을 층 단위로 잘라 여러 가속기에 싣고, 미니배치를 마이크로배치로 쪼개 파이프라인을 채우고, 기울기는 모아서 한 번에 적용하기 때문에 장치를 몇 대로 나누든 학습되는 수학은 한 장치에서 돌린 것과 같다는 점을 축으로 따라갑니다.

🔗 **[데모 바로가기](https://dev-jonghoonpark.github.io/how-gpipe-works/)**

## 다루는 내용

1. **한 장치에 안 들어가는 모델** · 표 1 Transformer의 파라미터 수 재구성(층당 50.36M + 131M), 파라미터당 12바이트, 16GB에 파라미터만으로는 25층까지 들어가지만 실제로는 활성 때문에 3층
2. **단순 모델 병렬과 유휴** · 층 사이 의존 때문에 한 번에 한 장치만 일함(그림 2b), 가동률 1/K
3. **마이크로배치 파이프라인** · K·M·역전파 비율을 바꾸며 재생하는 타임라인(그림 2c), 그 순간 일하는 장치 수
4. **버블 비율** · (K−1)/(M+K−1) 유도, M ≥ 4K 규칙, 표 2 측정값과 이상 모형 비교, (K−1)/M 표기와의 차이
5. **분할과 불균형** · 셀 비용 분산 최소화 DP와 층 수로 똑같이 자르기 비교, AmoebaNet이 선형으로 빨라지지 못한 이유
6. **재계산** · O(N × L) → O(N + (L/K)(N/M)), 장치 하나의 활성 메모리를 시간에 따라 추적, 시간 비용 +33%
7. **동기식 기울기 누적** · 브라우저에서 학습하는 MLP로 한 장치·GPipe·조각마다 갱신·지연된 기울기를 비교(GPipe와 한 장치의 가중치 차이 ~1e-15), BatchNorm 예외
8. **논문의 결과** · 표 1을 장치당 메모리로 다시 보기(83.9B, 298배), 표 3(고속 연결 없는 GPU), 표 4(AmoebaNet 84.4%와 전이 학습), 다국어 번역 6B 모델·깊이 대 너비·표 5
9. **다른 병렬화와 비교** · 데이터·텐서 병렬, PipeDream, GPipe 순서와 동기식 1F1B 타임라인 비교(같은 버블, 동시에 든 조각 M → K)
10. **퀴즈** 7문항

## 참고

- Huang et al. (2019), *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*, NeurIPS 2019 — [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
- Chen et al. (2016), *Training Deep Nets with Sublinear Memory Cost* — [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- Harlap et al. (2018), *PipeDream: Fast and Efficient Pipeline Parallel DNN Training* — [arXiv:1806.03377](https://arxiv.org/abs/1806.03377)
- Narayanan et al. (2021), *Memory-Efficient Pipeline-Parallel DNN Training*, ICML 2021

[how-ai-works](https://github.com/dev-jonghoonpark/how-ai-works) 시리즈의 일부입니다.
