# Mathematical derivations

역전파와 손실 함수의 수학적 유도를 노트북과 같은 내용으로 정리한다.
수식의 원본과 실행 코드는 아래 연결된 노트북에서 확인할 수 있다.

## Backpropagation

원본: [notebooks/backprop_derivation.ipynb](notebooks/backprop_derivation.ipynb)

### 2층 신경망 역전파: 수식 → 손계산 → NumPy 검증

입력 2 → 은닉 2 → 출력 1. 열벡터 관례로 계산한다.
원문에서 누락된 정답은 참고 예제와 동일하게 $y=1$로 명시한다.
모든 log는 자연로그, 배치 없이 단일 표본의 BCE를 사용한다.
자동미분은 사용하지 않는다.

#### 1. 고정 입력과 shape

| 기호 | shape | 값 |
|---|---|---|
| $x$ | (2,) | [1,0] |
| $W_1$ | (2,2) | [[0.1,0.2],[0.3,0.4]] |
| $b_1$ | (2,) | [0,0] |
| $W_2$ | (2,) | [0.5,0.6] |
| $b_2,y$ | scalar () | 0, 1 |

$W_1$의 행은 은닉 뉴런, 열은 입력 성분이다.
NumPy 1차원 배열에서는 전치보다 연산 관례와 outer를 명확히 적는다.

#### 2. 순전파 수식과 손계산

$$\sigma(z)=\frac{1}{1+e^{-z}},\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z)).$$

$$z_1=W_1x+b_1=
[0.1\cdot1+0.2\cdot0,\;0.3\cdot1+0.4\cdot0]
=[0.1,0.3]\quad(2,).$$

$$a_1=\sigma(z_1)=
[1/(1+e^{-0.1}),1/(1+e^{-0.3})]
\approx[0.5249791875,0.5744425168]\quad(2,).$$

$$z_2=W_2a_1+b_2
=0.5(0.5249791875)+0.6(0.5744425168)
\approx0.6071551038\quad().$$

$$\hat y=\sigma(z_2)\approx0.6472915700\quad().$$
$$L=-[y\log\hat y+(1-y)\log(1-\hat y)]
=-\log(0.6472915700)\approx0.4349584369\quad().$$

표시는 반올림하지만 다음 단계 계산에는 충분한 정밀도의 값을 유지한다.

#### 3. 연쇄 법칙을 출력에서 입력 방향으로 전개

**① 손실 → 예측 확률 (scalar)**
$$\frac{\partial L}{\partial\hat y}
=-\frac{y}{\hat y}+\frac{1-y}{1-\hat y}
=-\frac1{0.6472915700}\approx-1.5448988468\quad().$$

**② 예측 확률 → 출력 logit (scalar, 필수 체크포인트)**
$$\delta_2=\frac{\partial L}{\partial z_2}
=\frac{\partial L}{\partial\hat y}\hat y(1-\hat y)
=\hat y-y\approx-0.3527084300\quad().$$

**③ 출력 가중치와 편향**
$$\frac{\partial L}{\partial W_2}
=\delta_2a_1
\approx[-0.1851645850,-0.2026107182]\quad(2,),$$
$$\frac{\partial L}{\partial b_2}
=\delta_2\approx-0.3527084300\quad().$$

**④ 은닉 활성화로 전파**
$$\frac{\partial L}{\partial a_1}
=\delta_2W_2
\approx[-0.1763542150,-0.2116250580]\quad(2,).$$

**⑤ 은닉 logit으로 전파 (필수 체크포인트)**
$$\delta_1=\frac{\partial L}{\partial z_1}
=\frac{\partial L}{\partial a_1}\odot a_1\odot(1-a_1).$$
$$\delta_{1,1}=(-0.1763542150)(0.5249791875)(0.4750208125)
\approx-0.0439785158,$$
$$\delta_{1,2}=(-0.2116250580)(0.5744425168)(0.4255574832)
\approx-0.0517335044.$$
$$\delta_1\approx[-0.0439785158,-0.0517335044]\quad(2,).$$

**⑥ 입력 가중치와 편향**
$$\frac{\partial L}{\partial W_1}
=\delta_1x^T
=\operatorname{outer}(\delta_1,x)
\approx\begin{bmatrix}-0.0439785158&0\\-0.0517335044&0\end{bmatrix}
\quad(2,2),$$
$$\frac{\partial L}{\partial b_1}=\delta_1\quad(2,).$$

추가로 $\partial L/\partial x=W_1^T\delta_1$이며 shape은 (2,)이다.
두 번째 입력이 0이므로 $W_1$ 두 번째 열의 기울기가 0인 점도 확인한다.
원문 참고값 -0.0513은 계산 오류이며 올바른 4자리 값은 -0.0517이다.

#### 4. 독립적으로 적은 손계산 상수와 소수점 4자리 비교

아래 기대값은 수식에 숫자를 대입해 얻은 상수다.
코드 출력에서 기대값을 재생성하지 않는다.
모든 순전파 중간값과 필수 역전파 항목, 편향을 비교한다.

| 항목 | shape | 손계산 (4자리) | NumPy (4자리) | 일치 |
|---|---|---|---|---|
| z1 | (2,) | [0.1000 0.3000] | [0.1000 0.3000] | PASS |
| a1 | (2,) | [0.5250 0.5744] | [0.5250 0.5744] | PASS |
| z2 | () | 0.6072 | 0.6072 | PASS |
| y_pred | () | 0.6473 | 0.6473 | PASS |
| loss | () | 0.4350 | 0.4350 | PASS |
| d_y_pred | () | -1.5449 | -1.5449 | PASS |
| d_z2 | () | -0.3527 | -0.3527 | PASS |
| W2 | (2,) | [-0.1852 -0.2026] | [-0.1852 -0.2026] | PASS |
| b2 | () | -0.3527 | -0.3527 | PASS |
| d_a1 | (2,) | [-0.1764 -0.2116] | [-0.1764 -0.2116] | PASS |
| d_z1 | (2,) | [-0.0440 -0.0517] | [-0.0440 -0.0517] | PASS |
| W1 | (2, 2) | [[-0.0440  0.0000]  [-0.0517  0.0000]] | [[-0.0440 0.0000]  [-0.0517 0.0000]] | PASS |
| b1 | (2,) | [-0.0440 -0.0517] | [-0.0440 -0.0517] | PASS |

#### 5. 중심차분으로 모든 학습 파라미터를 별도로 검증

$$\frac{\partial L}{\partial\theta_i}
\approx\frac{L(\theta+h e_i)-L(\theta-h e_i)}{2h},
\qquad h=10^{-5}.$$

이는 독립적인 검증이며 역전파 구현을 대신하지 않는다.
float64에서 모든 파라미터의 최대 절대오차가 $10^{-7}$ 이하여야 한다.

#### 해석

이 표본에서 출력 확률은 정답 1보다 작다. 출력 logit의 기울기가 음수이므로
경사하강법은 출력 logit을 키우는 방향으로 파라미터를 움직인다.
첫 번째 입력만 활성화되어 W1의 첫 번째 열만 직접 업데이트된다.
각 층은 행렬 연산을 하고 역전파는 이 연산들의 국소 미분을 연쇄 법칙으로 연결한다.

## Probability and loss

원본: [notebooks/probability_loss.ipynb](notebooks/probability_loss.ipynb)

### 확률 분포와 손실 함수의 수학적 연결

정규분포, 베르누이, 소프트맥스, MLE, 정보 이론을 NumPy로 확인한다.
$N(\mu,\sigma^2)$ 관례를 사용하므로 $N(2,0.5)$의
분산은 0.5, 표준편차는 $\sqrt{0.5}$이다. 모든 log는 자연로그다.

#### 1. 정규분포 PDF와 베르누이 PMF
$$p(x|\mu,\sigma^2)=\frac1{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right].$$
연속분포의 확률은 PDF의 구간 적분이며 한 점에서의 높이가 아니다.
$$P(X=x)=p^x(1-p)^{1-x},\qquad x\in\{0,1\}.$$

#### 2. 소프트맥스와 수치 안정성
$$q_k=\frac{\exp(z_k-m)}{\sum_j\exp(z_j-m)},\qquad m=\max_jz_j.$$
같은 상수 $m$을 빼도 비율은 같고 큰 logit에서 overflow를 피한다.
softmax는 비음수 확률을 만들고 총합을 1로 정규화한다.

#### 3. MSE 최소화 = 정규 잡음 가정 아래의 MLE

가정: 관측치는 조건부 독립이고
$y_i=f_\theta(x_i)+\epsilon_i$,
$\epsilon_i\sim N(0,\sigma^2)$이다.
$\sigma^2>0$는 모든 표본에서 같고 $\theta$와 무관한 고정값이다.

$$\mathcal L(\theta)=\prod_{i=1}^n
\frac1{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\right].$$

$$\log\mathcal L(\theta)
=-\frac n2\log(2\pi\sigma^2)
-\frac1{2\sigma^2}\sum_i(y_i-f_\theta(x_i))^2.$$

따라서
$$\arg\max_\theta\mathcal L
=\arg\min_\theta[-\log\mathcal L]
=\arg\min_\theta\sum_i(y_i-f_\theta(x_i))^2
=\arg\min_\theta\frac1n\sum_i(y_i-f_\theta(x_i))^2.$$

고정된 양의 배율과 $\theta$에 무관한 상수를 제거했으므로 최적점이 같다.
**MSE와 NLL의 숫자 자체가 같은 것은 아니다.**
분산이 표본마다 다르면 가중 제곱오차가 되고,
분산도 모델이 예측하면 log 분산 항을 버릴 수 없다.

#### 4. Binary Cross-Entropy 최소화 = 베르누이 MLE

$y_i\in\{0,1\}$, $p_i=f_\theta(x_i)\in(0,1)$라 하자.
$$P(y_i|x_i;\theta)=p_i^{y_i}(1-p_i)^{1-y_i}.$$
조건부 독립 표본의 log likelihood는
$$\log\mathcal L(\theta)=
\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].$$
$$-\frac1n\log\mathcal L(\theta)
=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]
=\operatorname{BCE}.$$
즉 베르누이 확률의 최대우도 추정과 평균 BCE 최소화는 동일한 최적점이다.

#### 5. Categorical Cross-Entropy 최소화 = 카테고리 MLE

$y_{ik}$는 정답의 one-hot 성분이고 $q_{ik}$는 softmax 확률이다.
$$P(y_i|x_i;\theta)=\prod_{k=1}^K q_{ik}^{y_{ik}},$$
$$-\frac1n\log\mathcal L(\theta)
=-\frac1n\sum_i\sum_k y_{ik}\log q_{ik}
=\operatorname{CategoricalCE}.$$

회귀의 연속 오차에는 정규분포, 이진 분류에는 베르누이,
상호 배타적인 다중 분류에는 카테고리 분포를 사용한다.

#### 6. 보너스: Entropy, KL-Divergence, Cross-Entropy

$$H(p)=-\sum_kp_k\log p_k,$$
$$D_{KL}(p\Vert q)=\sum_kp_k\log\frac{p_k}{q_k},$$
$$H(p,q)=-\sum_kp_k\log q_k=H(p)+D_{KL}(p\Vert q).$$

$0\log0=0$으로 정의한다. $p_k>0$인데 $q_k=0$이면
Cross-Entropy와 KL은 무한대다. 단위는 nats이며 KL은 비대칭이다.
데이터의 정답 분포 $p$가 고정되면 Cross-Entropy를 줄이는 것은
예측 분포와의 KL을 줄이는 것과 같다.

## Source files

- [역전파 구현](src/backprop.py)
- [확률 분포 및 손실 관련 함수](src/probability.py)
- [고유값의 직접 계산 및 참조 비교](src/linear_algebra.py)
- [전체 수치 검증 실행](scripts/run_all.py)
