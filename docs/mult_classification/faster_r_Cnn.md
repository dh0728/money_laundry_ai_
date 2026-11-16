# Faster R-CNN 리뷰

## R-CNN

### R-CNN이 무엇인가?

이미지 분류는 "이 사진은 고양이다"라고 답 하나만 내면 된다. 객체 탐지는 "여기(박스)에 사람, 저기(박스)에 말"처럼 위치와 종류를 둘 다 맞춰야 한다.

가장 단순한 방법은 모든 위치와 크기를 모두 잘라서 CNN에 넣어보는 것인데, 이건 경우의 수가 너무 많아서 현실적으로 불가능하다.

그래서 R-CNN은 일을 두 단계로 나눈다.

1. 물체가 있을 법한 영역만 먼저 골라낸다. **(Region Proposal)**
2. 고른 영역만 CNN으로 판별한다.

### Selective Search

1단계에서 쓰는 방법이 **Selective Search**이다. 학습하는 모델이 아니라 고전적인 영상처리 알고리즘이고, 동작 방식은 이렇다.

1. 이미지를 색과 질감이 비슷한 작은 조각(superpixel) 수백 개로 쪼갠다.
2. 비슷한 이웃 조각끼리 계속 합쳐 나간다.
3. 합쳐지는 과정에서 생긴 덩어리마다 박스를 쳐서 후보를 낸다.

물체는 보통 색이나 질감이 비슷한 영역으로 뭉쳐 있으니, 이런 덩어리가 물체일 가능성이 높다는 것에서 나온 발상이다. R-CNN은 이렇게 나온 후보 중 약 2000개를 쓴다.

### Warped region

후보 박스들은 크기와 비율이 제각각이다. 그런데 당시 CNN(AlexNet)은 마지막 FC 층 때문에 입력 크기가 227×227로 고정이었다. 그래서 각 후보 영역을 잘라낸 뒤 비율을 무시하고 억지로 227×227로 늘리거나 줄인다. 이것이 **warped region**이다.

### R-CNN 처리 흐름

그다음 흐름은 다음과 같다.

```text
후보 영역 1 → 227×227로 warp → CNN → 특징 벡터 → SVM: "사람? yes"
후보 영역 2 → 227×227로 warp → CNN → 특징 벡터 → SVM: "비행기? no"
...
후보 영역 2000 → ...
```

여기에 박스 위치를 미세 조정하는 **bounding box regression**도 붙는다.

### R-CNN의 병목

병목이 생기는 이유는 **이미지 한 장에 CNN을 2000번 돌린다**는 점이다.

- CNN의 순전파 한 번은 비싼 연산에 속하는데, 이걸 이미지 한 장당 2000번 반복한다.
- 게다가 후보 박스들은 서로 많이 겹친다. 말 머리를 포함한 박스가 수십 개라면, 같은 픽셀에 대한 합성곱을 수십 번 중복 계산하는 셈이다.
- 그 결과 이미지 한 장을 처리하는 데 GPU로도 수십 초가 걸린다. 실시간 처리가 불가능하다.

## Fast R-CNN

이런 R-CNN의 CNN에서 병목 현상을 줄이기 위해 개발된 시스템이 Fast R-CNN이다. R-CNN과 다르게 Fast R-CNN의 이미지는 한 번만 CNN을 거친다.

### 핵심 아이디어

Fast R-CNN은 CNN의 순서를 뒤집어 **이미지 전체를 CNN에 한 번에 통과 → 그 결과물(feature map)에서 잘라낸다.**

이게 가능한 이유는 CNN의 합성곱층이 **위치 정보를 유지**하기 때문이다.

- feature map은 이미지를 작게 축소한 지도와 같아서, 원본의 객체의 위치 정보가 feature map에도 비슷한 위치에 담겨 있다.
- 그러니 원본에서 박스를 잘라 CNN에 다시 넣을 필요 없이 feature map에서 해당 위치를 잘라 쓰면 된다.
- 겹치는 영역의 합성곱을 다시 계산하는 중복도 사라진다.

### Fast R-CNN 전체 흐름

전체 흐름은 다음과 같고, 두 갈래로 따로 돌다가 합쳐진다.

```text
                ┌─ (GPU) CNN 1회 ──────────▶ feature map 1개
원본 이미지 ────┤                                   │
                └─ (CPU) Selective Search ─▶ 후보 박스 2,000개
                                                    │
                                     박스 좌표를 feature map에 투영 (RoI projection)
                                                    ▼
                                     RoI Pooling → 고정 크기(예: 7×7)
                                                    ▼
                                     FC층 → softmax(클래스) + bbox regressor(위치)
```

> Selective Search는 여전히 원본 이미지 위에서 따로 돌아간다는 점이 중요하다. CNN과 별개의 과정이다.

### RoI Projection(투영)

feature map은 원본보다 작다. VGG16 기준으로 마지막 합성곱 층까지 가로, 세로가 1/16로 줄어든다(stride 16). 그래서 원본에서 구한 박스 좌표를 16으로 나눠 feature map 위의 좌표로 바꾼다.

**예시**

- 원본 이미지 800×600 → feature map 50×37 (채널 512개)
- 원본의 후보 박스 (x1=160, y1=96, x2=480, y2=416)
- feature map 위 좌표: 각각 ÷16 → (10, 6, 30, 26)

이렇게 feature map에서 그 영역을 떼어낸 것을 **RoI(Region of Interest)**라고 부르고, 슬라이드의 "RoI projection" 화살표가 이 과정이다.

### RoI Pooling

떼어낸 RoI도 크기가 제각각인데(위 예시는 20×20, 다른 박스는 5×12일 수도 있다), 뒤의 FC층은 입력 크기가 고정이어야 한다.

- **R-CNN**: 이미지를 억지로 늘려서(warp) 맞췄다.
- **Fast R-CNN**: RoI Pooling을 쓴다.

RoI Pooling은 다음과 같이 동작한다.

1. RoI를 크기와 상관없이 7×7 격자로 나눈다.
2. 각 칸에서 최댓값 하나만 뽑는다(max pooling).
3. 결과는 항상 7×7×512이다.

이 뒷부분(RoI Pooling → FC → 분류·회귀)은 RoI마다 반복되지만, 비싼 합성곱 연산은 이미지당 한 번만 일어난다.

### Fast R-CNN의 병목: Selective Search

CNN 부분은 이렇게 빨라졌는데, 후보 생성은 그대로 Selective Search이다. 이게 GPU에서 돌지 않는 이유는 연산 성격이 다르기 때문이다.

- **GPU가 잘하는 것**: 같은 계산(곱하고 더하기)을 수많은 데이터에 동시에 하는 일. 합성곱이 딱 이런 구조이다.
- **Selective Search가 하는 것**: "지금 가장 비슷한 두 영역을 찾아 합치고, 유사도를 다시 계산하고, 또 가장 비슷한 쌍을 찾고..."를 반복한다. 앞 단계 결과에 따라 다음 단계가 정해지는 순차적이고 분기가 많은 알고리즘이라 병렬화가 어렵다. 그래서 CPU에서 돌아가도록 구현되어 있다.

## Faster R-CNN
Fast R-CNN의 Selective search 알고리즘을 수행하면서 생기는 병목 현상을 해결하기 위해 나온 것이 Faster R-CNN이다. Faster R-CNN에서의 가장 큰 특징은 바로 RPN이다. RPN은 Region proposal을 수행해주는 CNN 구조의 네트워크이다. 
RPN은 CPU보다 계산 속도가 빠른 GPU를 사용할 수 있어서 이전에 Region proposal을 수행하기 위해 CPU를 사용했던 Fast R-CNN보다 속도가 빠르다. 

### Faster R-CNN 구조

Faster R-CNN은 크게 3개의 모듈로 볼 수 있다. 

#### Backbone 네트워크
- Backbone 네트워크는 입력 이미지를 feature map으로 변환하는 네트워크
  - 이 논문에서는 ZF와 VGG-16 모델 사용 

#### RPN(Region Proposal Network) 모듈
- Feature map으로부터 region proposal 수행
- Fast R-CNN이 어디를 봐야 할지 알려줌

#### Fast R-CNN의 Detection 모듈
- Rol pooling 수행
  - 각각의 사이즈가 다른 Region proposal에 대해서 동일한 크기의 Region proposal로 변환을 해주는 레이어
- 제안된 영역에 대해 classification과 Bounding-box regression을 수행
  - Bounding-box regression : 후보 박스의 위치와 크기를 정답 박스에 가깝게 미세 조정하는 것

### RPN의 동작방식
1. Backbone 네트워크가 출력한 feature map이 RPN의 input으로 들어감
2. feature map 위에 3 x 3 convolution을 적용 (= 3 x 3 sliding window로 모든 위치를 훑는 것과 같음)
3. 각 위치마다 중간 특징 벡터가 나옴. 차원은 backbone의 마지막 채널 수를 따름
   - ZF(ZFNet, 2013 ILSVRC 우승, 합성곱 5층): 256-d
   - VGG-16: 512-d
4. feature map의 각 위치마다 미리 정해 둔 anchor box k개(k=9)를 원본 이미지 위에 둠
   - 크기 3종(128², 256², 512²) x 비율 3종(1:1, 1:2, 2:1) = 9개
   - anchor는 네트워크가 만드는 것이 아니라 고정된 기준 박스
5. 중간 특징 벡터는 두 갈래(각각 1 x 1 conv)로 나뉨. 같은 특징을 공유하는 sibling layer
   - cls layer: anchor마다 물체/배경 점수 2개 → 2k = 18개 출력. softmax 후 물체일 확률
   - reg layer: anchor마다 보정값 4개(tx, ty, tw, th) → 4k = 36개 출력
     - 중심 x, y 이동량과 너비, 높이 배율. 좌표 자체가 아니라 anchor 대비 offset
6. 보정값을 anchor에 적용해 박스를 만들고, 물체 확률 순 정렬 + NMS로 겹침 제거
   → 상위 박스들(테스트 시 약 300개)을 region proposal로 detection 모듈에 전달

#### Anchor box
- feature map의 각 위치를 중심으로 원본 이미지 위에 깔아 두는 크기·비율이 다른 기준 박스
- 다양한 크기·모양의 물체를 잡기 위해 여러 종류를 미리 정해 둠
- RPN은 각 anchor에 대해 "물체가 있는가"와 "박스를 어떻게 고칠까"만 학습
- Anchor box 중에 feature map의 경계 밖으로 넘어가는 경우는 무시

### Faster R-CNN 전체 흐름
1. Backbone(VGG-16)이 원본 이미지(H x W)를 feature map(h x w x 512, 약 1/16 크기)으로 변환
   - 이 feature map은 한 번만 계산하고 RPN과 detection 모듈이 함께 사용 (연산 공유)
2. RPN이 feature map으로부터 region proposal 생성
   - 3 x 3 conv(padding으로 h x w 유지) → cls(h x w x 2k), reg(h x w x 4k)
   - anchor h x w x k개에 보정값 적용 → 물체 확률 순 정렬 → NMS(IoU 0.7)
   - 상위 proposal(테스트 시 약 300개)만 다음 단계로 전달
3. RoI Pooling: proposal 좌표를 feature map 위에 투영해 해당 영역을 잘라냄
   - 크기가 제각각인 영역을 고정 크기(l x l, VGG 기준 7 x 7)로 max pooling
4. Fast R-CNN 헤드: FC층을 거쳐 두 갈래로 출력
   - 분류: K개 클래스 + 배경 → K+1개 확률
   - 박스 보정: 클래스별 4개 → 4K개 값
5. NMS로 중복 박스 제거 후 최종 예측 출력

### NMS (Non-Maximum Suppression)
- 같은 물체에 중복으로 찍힌 박스들 중 가장 점수가 높은 것만 남기는 알고리즘
  - 이미지 전체에서 하나만 남기는 것이 아님. 물체가 여러 개면 박스도 여러 개 남음
- 동작 (클래스별로 수행)
  1. 점수 높은 순으로 정렬
  2. 최고점 박스를 확정
  3. 확정 박스와 IoU가 기준값 이상인 박스를 제거
  4. 남은 박스로 2~3 반복
- IoU = 겹친 넓이 / 합친 넓이 (0: 안 겹침, 1: 완전히 일치)
- Faster R-CNN에서는 두 번 사용: RPN 내부(proposal 정리), 최종 출력(결과 정리)

### Faster R-CNN의 특성 - Translation-Invariant
이미지속 객체의 위치가 달라져도 동일한 객체로 인식하는 특성
- Faster R-CNN은 Translation-Invariant하지 않은 Object Detection 시스템에 비해서 모델의 사이즈가 작다는 장점도 있음
- 작은 데이터 셋을 학습할 때 overfitting될 확률이 적다.

---

## 출처

- [R-CNN 계열의 발전 동향과 Faster R-CNN 논문 리뷰](https://medium.com/@parkie0517/r-cnn-%EA%B3%84%EC%97%B4%EC%9D%98-%EB%B0%9C%EC%A0%84-%EB%8F%99%ED%96%A5%EA%B3%BC-faster-r-cnn-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-155f46cca6ad) - Medium (@parkie0517)
