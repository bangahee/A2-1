# 노트북 셀 위치 안내

셀 순서는 1부터, cell-NNN ID는 0부터 센다. Code 번호는 실행 순서다.

## [backprop_derivation.ipynb](../notebooks/backprop_derivation.ipynb)

| 셀 순서 | 고정 셀 ID | 구분 | 내용 |
|---|---|---|---|
| 1 | cell-000 | Markdown | 2층 신경망 역전파: 수식 → 손계산 → NumPy 검증 |
| 2 | cell-001 | Code 1 | from pathlib import Path |
| 3 | cell-002 | Markdown | 1. 고정 입력과 shape / 2. 순전파 수식과 손계산 |
| 4 | cell-003 | Markdown | 3. 연쇄 법칙을 출력에서 입력 방향으로 전개 |
| 5 | cell-004 | Code 2 | from src.backprop import fixed_example, forward, backward |
| 6 | cell-005 | Markdown | 4. 독립적으로 적은 손계산 상수와 소수점 4자리 비교 |
| 7 | cell-006 | Code 3 | hand_forward = { |
| 8 | cell-007 | Markdown | 5. 중심차분으로 모든 학습 파라미터를 별도로 검증 |
| 9 | cell-008 | Code 4 | from src.calculus import numerical_gradient |
| 10 | cell-009 | Markdown | 해석 |

## [experiment_validation.ipynb](../notebooks/experiment_validation.ipynb)

| 셀 순서 | 고정 셀 ID | 구분 | 내용 |
|---|---|---|---|
| 1 | cell-000 | Markdown | Experiment validation · 재현 조건과 실험 결과 |
| 2 | cell-001 | Code 1 | from pathlib import Path |
| 3 | cell-002 | Markdown | 1. PNG manifest and SVD image size |
| 4 | cell-003 | Code 2 | display_table(["출력 경로", "해상도 (px)", "dpi"], [ |
| 5 | cell-004 | Markdown | 2. Power Iteration vs eig |
| 6 | cell-005 | Code 3 | from src.linear_algebra import power_iteration |
| 7 | cell-006 | Markdown | 3. Central differences: h study |
| 8 | cell-007 | Code 4 | from scripts.experiment_evidence import derivative_step_study |
| 9 | cell-008 | Markdown | 4. Optimizer hyperparameters and fixed initial conditions |
| 10 | cell-009 | Code 5 | from src.optimizer import VanillaGD, optimize, quadratic_gradient |
| 11 | cell-010 | Markdown | 5. Divergence verification for lr >= 0.5 |
| 12 | cell-011 | Code 6 | from scripts.experiment_evidence import divergence_study |

## [probability_loss.ipynb](../notebooks/probability_loss.ipynb)

| 셀 순서 | 고정 셀 ID | 구분 | 내용 |
|---|---|---|---|
| 1 | cell-000 | Markdown | 확률 분포와 손실 함수의 수학적 연결 |
| 2 | cell-001 | Code 1 | from pathlib import Path |
| 3 | cell-002 | Markdown | 1. 정규분포 PDF와 베르누이 PMF |
| 4 | cell-003 | Code 2 | from src.probability import ( |
| 5 | cell-004 | Markdown | 2. 소프트맥스와 수치 안정성 |
| 6 | cell-005 | Code 3 | logits = np.array([1000., 1001., 1002.]) |
| 7 | cell-006 | Markdown | 3. MSE 최소화 = 정규 잡음 가정 아래의 MLE |
| 8 | cell-007 | Code 4 | observed = np.array([0.2, 0.8, 1.4]) |
| 9 | cell-008 | Markdown | 4. Binary Cross-Entropy 최소화 = 베르누이 MLE / 5. Categorical Cross-Entropy 최소화 = 카테고리 MLE |
| 10 | cell-009 | Code 5 | targets = np.array([0., 1., 1.]) |
| 11 | cell-010 | Markdown | 6. 보너스: Entropy, KL-Divergence, Cross-Entropy |
| 12 | cell-011 | Code 6 | p, q = np.array([.3, .7]), np.array([.6, .4]) |
