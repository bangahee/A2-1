# A2-1 제출 요구사항 점검표

모든 수치 기준은 outputs/metrics.json, 검증 코드는 scripts/run_all.py를 참조한다.
두 노트북은 결과를 저장한 실행본이며 GitHub에서 코드와 출력 모두 읽을 수 있다.

| 요구사항 | 구현/증거 | 검증 기준 또는 처리 |
|---|---|---|
| 회전 행렬과 단위 원의 전후 겹침 | src/linear_algebra.py, outputs/matrix_transformations.png | R(π/4), 같은 축척 |
| S(2,0.5)와 타원 | 같은 모듈/그림 | 가로 2배, 세로 0.5배 |
| Sh(k)와 기울어진 형태 | 같은 모듈/그림 | k=0.8 |
| 행렬식과 면적비 | area_report, metrics.areas | 상대오차 ≤1%, 음의 det는 절댓값 |
| 직접 Power Iteration | power_iteration | 행렬-벡터 곱, 정규화, Rayleigh quotient |
| eig 검증 전용 사용 | scripts/run_all.py | 고유값 상대오차 ≤5% |
| SVD k=10,50,100 | outputs/svd_comparison.png | 요청값 모두 표시, 유효 k=10,50,64 |
| 64×64 이하 회색조 이미지 | data/grace_hopper_64.npy, data/README.md | 공개 예제 출처와 전처리 명시 |
| 중심차분 f=x², x=3 | src/calculus.py, metrics.derivative | 절대오차 ≤1e-4 |
| f=x²+y² 등고선과 gradient 화살표 | outputs/gradient_contours.png | 여러 점, 동일 축척 |
| 등고선 수직 확인 | 같은 그림, gradient_tangent_dot_product | (2x,2y)·(-y,x)=0 |
| 2→2→1 순전파/역전파 수식 | notebooks/backprop_derivation.ipynb | 출력에서 입력 방향의 연쇄 법칙 |
| 고정 숫자 예제와 손계산 | 같은 노트북 | x,W1,b1,W2,b2 그대로 사용, y=1 가정 |
| z1,a1,z2,y_pred 기록 | 같은 노트북 | 손계산 상수와 코드 출력 모두 포함 |
| 필수 6개 미분 항목 | 같은 노트북 | dL/dp,dL/dz2,dL/dW2,dL/da1,dL/dz1,dL/dW1 |
| 모든 단계 shape | 같은 노트북 | scalar (), (2,), (2,2) 구별 |
| 손계산과 NumPy 4자리 일치 | 같은 노트북의 assert_array_equal | 편향/손실 추가, 중심차분 교차 검증 |
| VanillaGD 클래스 | src/optimizer.py | θ←θ-lr∇f |
| lr=0.1,100회,반경0.1 이내 | metrics.circle | 초기점 (5,5), 최종 반경 약 1.44e-9 |
| 등고선 위 점선 수렴 경로 | outputs/gd_momentum_circle.png | 초기점 포함 전체 101개 점 |
| 큰 lr의 발산 | outputs/learning_rate_stability.png | 1.1 발산; 0.5 이상이라는 원문의 오류 설명 |
| Momentum β=0.9 | src/optimizer.py | 두 함수에서 GD와 같은 Figure 비교 |
| 타원형 GD/Momentum 비교 | outputs/gd_momentum_ellipse.png | lr=0.09에서의 실제 결과 |
| 타원형 Momentum 이점 확인 | outputs/bonus_optimizer_paths.png, bonus_optimizer_convergence.png | 같은 lr=0.01에서 최초 진입 194→37회 |
| Normal PDF 2종 | notebooks/probability_loss.ipynb, outputs/probability_distributions.png | N(0,1),N(2,0.5), 분산 관례 |
| Bernoulli PMF 2종 | 같은 노트북/그림 | p=0.3,0.7 |
| 직접 softmax와 합 검증 | src/probability.py, 같은 노트북 | 합과 1의 차이 ≤1e-6, 큰 logit 안정성 |
| MSE–MLE 유도 | 같은 노트북 | Gaussian likelihood→log→상수제거→MSE |
| CE–MLE 유도 | 같은 노트북 | Bernoulli와 categorical 각각 유도 |
| README 개요/실행/주요결과 | README.md | 처음부터 실행할 명령과 기대 출력 |
| NumPy 구현, 시각화 라이브러리 제한 | src/*.py, tests/test_mathematics.py | 자동미분 및 sklearn 없음 |
| 시드 42 | run_all.py와 두 노트북 | 결정적인 초기값/실험 |
| 함수/클래스 docstring | src/*.py | AST 검사 포함 |
| 수식 설명 | 노트북, 코드 docstring, docs | 연산과 수학 연결 |
| 라이브러리/버전 | requirements.txt, requirements-dev.txt | Python≥3.12, 실검증 3.14.5 |
| PNG는 outputs | outputs/*.png | 12개 figure |
| GitHub 레포 URL | https://github.com/bangahee/A2-1 | main이 제출 브랜치 |
| 보너스 Adam | src/optimizer.py, bonus_optimizer_*.png | 편향 보정, 3개 방법 비교 |
| 보너스 Newton | src/optimizer.py, outputs/bonus_newton.png | Hessian 사용, 한 단계 원점 |
| 보너스 정보 이론 | src/probability.py, probability_loss.ipynb | Entropy,KL,CE 직접 구현 및 항등식 |

## 실행으로 확인하는 제출 상태

1. python -m unittest discover -s tests -v → 39 tests, OK.
2. python -m scripts.run_all → 16개 수치 허용 기준 PASS.
3. python -m scripts.execute_notebooks → 두 노트북의 총 10개 code cell PASS.
4. python -m scripts.verify_submission → 12개 PNG와 실행본 구조 PASS.
5. python -m scripts.check_conventions → 커밋 형식과 브랜치 이름 PASS.

원문의 모순과 수정 이유는 [MATHEMATICAL_DECISIONS.md](MATHEMATICAL_DECISIONS.md)를 읽는다.
