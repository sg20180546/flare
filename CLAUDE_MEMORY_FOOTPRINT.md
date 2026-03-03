# FLARE Memory Footprint 분석

**기준 설정**: RTX A6000 논문 환경, **P=6** (병렬 환경 수), float32 (4 bytes/param)
**`num_sem_categories`**: `3 + 102×P = 3 + 102×6 = 615` → 채널 수 = 616 (1+615)

---

## 모듈별 분석

---

### 1. Mask R-CNN × 3 모델 (`models/segmentation/`)

세 모델이 **동시에 GPU 메모리에 올라감**:
- `sem_seg_model_alfw_large`: receptacle 감지 (33 classes)
- `sem_seg_model_alfw_small`: pickupable 물체 감지 (75 classes)
- `sem_seg_model_moca`: MOCA pretrained (119 classes)

#### 공유 구조 파라미터 (모델별 독립 로드)

**ResNet-50 Backbone**
```
conv1 (3→64, k=7):            64 × (3×7×7 + 1)  =      9,472
layer1 (3× Bottleneck 64→256): ≈ 256K
layer2 (4× Bottleneck 128→512): ≈ 1,117K
layer3 (6× Bottleneck 256→1024): ≈ 6,427K
layer4 (3× Bottleneck 512→2048): ≈ 13,902K
ResNet-50 소계: 25,557,032 params
```

**FPN Neck** (C2~C5 → 256ch 통일)
```
lateral connections (1×1): 65,792 + 131,328 + 262,400 + 524,544
output projections (3×3):  590,080 × 4
FPN 소계: 3,344,192 params
```

**box_head (TwoMLPHead, 공통)**
```
RoIAlign 출력: 256 × 7 × 7 = 12,544
fc6: 12,544 × 1024 + 1024 = 12,846,080
fc7:  1,024 × 1024 + 1024 =  1,049,600
box_head 소계: 13,895,680 params  ← 모델당 가장 큰 단일 블록
```

**Backbone+FPN+box_head 공통 합계: 42,796,904 params**

---

#### Model 1: `alfw_large` (33 classes, 커스텀 RPN 32 anchors)

```
RPN (anchor=32): conv(590,080) + cls(8,224) + bbox(32,896)  =     631,200
FastRCNNPredictor(1024, 33):
  cls_score: 1024 × 33 + 33                               =      33,825
  bbox_pred: 1024 × 132 + 132                              =     135,300
MaskRCNNPredictor(256, 256, 33):
  4× conv(256→256): 590,080 × 4                            =   2,360,320
  conv_transpose: 256 × (256×4 + 1)                        =     262,400
  mask_fcn_logits: 33 × (256 + 1)                          =       8,481
  소계: 2,631,201

총: 42,796,904 + 631,200 + 169,125 + 2,631,201 = 46,228,430 params
메모리: 46,228,430 × 4 = 184.9 MB ≈ 0.172 GB
```

#### Model 2: `alfw_small` (75 classes, 커스텀 RPN 32 anchors)

```
RPN: 631,200 (동일)
FastRCNNPredictor(1024, 75):
  cls_score: 1024 × 75 + 75                               =      76,875
  bbox_pred: 1024 × 300 + 300                              =     307,500
MaskRCNNPredictor(256, 256, 75):
  공통: 2,622,720
  mask_fcn_logits: 75 × 257                                =      19,275
  소계: 2,641,995

총: 42,796,904 + 631,200 + 384,375 + 2,641,995 = 46,454,474 params
메모리: 46,454,474 × 4 = 185.8 MB ≈ 0.173 GB
```

#### Model 3: `moca` (119 classes, 기본 RPN 9 anchors)

```
RPN (anchor=9): conv(590,080) + cls(2,313) + bbox(9,252)   =     601,645
FastRCNNPredictor(1024, 119):
  cls_score: 1024 × 119 + 119                              =     121,975
  bbox_pred: 1024 × 476 + 476                              =     487,900
MaskRCNNPredictor(256, 256, 119):
  공통: 2,622,720
  mask_fcn_logits: 119 × 257                               =      30,583
  소계: 2,653,303

총: 42,796,904 + 601,645 + 609,875 + 2,653,303 = 46,661,727 params
메모리: 46,661,727 × 4 = 186.6 MB ≈ 0.174 GB
```

#### Mask R-CNN 3모델 합계

| 모델 | 파라미터 수 | 가중치 메모리 |
|------|-----------|-------------|
| alfw_large (33 cls) | 46,228,430 | 0.172 GB |
| alfw_small (75 cls) | 46,454,474 | 0.173 GB |
| moca (119 cls) | 46,661,727 | 0.174 GB |
| **소계** | **139,344,631 (139.3M)** | **0.519 GB** |

> 추론 중 중간 feature map 활성화 메모리: 입력 300×300 이미지 기준 FPN 다단계 feature가 약 0.3 GB/모델 → **+0.9 GB** (3모델 합산, 추론 시 순차 실행 시 약 0.3 GB)

---

### 2. UNetMulti — Semantic Policy (`models/semantic_policy/sem_map_model.py`)

```python
UNetMulti(num_sem_categories=24)  # main.py 줄 100
입력 채널 = num_sem_categories + 4 = 28
```

**파라미터 계산** (`Conv2d: out × (in × k² + 1)`)

| 레이어 | 구성 | 파라미터 수 |
|--------|------|-----------|
| Conv1 | 28→32, k=3, p=2 | 32 × (28×9+1) = **8,096** |
| Conv2 | 32→64, k=3, p=2 | 64 × (32×9+1) = **18,496** |
| Conv3 | 64→256, k=3, p=2 | 256 × (64×9+1) = **147,712** |
| Conv4 | 256→256, k=3, p=1 | 256 × (256×9+1) = **590,080** |
| Conv5 | 256→128, k=3, p=1 | 128 × (256×9+1) = **295,040** |
| Conv6 | 128→128, k=3, p=1 | 128 × (128×9+1) = **147,584** |
| Conv7 | 128→73, k=3, p=1 | 73 × (128×9+1) = **84,169** |
| **합계** | | **1,291,177** |

```
총 파라미터: 1,291,177 ≈ 1.29M
메모리: 1,291,177 × 4 = 5.17 MB ≈ 0.0048 GB
```

입력 텐서 (240×240 맵 × 28채널 × 6 batch):
```
6 × 28 × 240 × 240 × 4 = 38,707,200 bytes ≈ 36.9 MB
```

---

### 3. VALTS Depth 모델 — SimpleUNet (`models/depth/alfred_perception_models.py`)

```python
# segmentation_definitions.py: get_num_objects() = 129
# depth_bins = 50
num_c = 129
out_channels = 129 + 50 = 179
hc1 = hc2 = 256
```

**DoubleConv(cin→cout)** = `Conv2d(cin,cin)` + `Conv2d(cin,cout)`:
- params = `cin×(cin×9+1) + cout×(cin×9+1)`

**UpscaleDoubleConv(cin→cout)** = `Conv2d(cin,cout)` + `Conv2d(cout,cout)`:
- params = `cout×(cin×9+1) + cout×(cout×9+1)`

| 블록 | 구성 | 파라미터 수 |
|------|------|-----------|
| conv1 | DoubleConv(3→256) | 3×(3×9+1) + 256×(3×9+1) = 84 + 7,168 = **7,252** |
| conv2 | DoubleConv(256→256) | 256×(256×9+1) × 2 = **1,180,160** |
| conv3 | DoubleConv(256→256) | 동일 = **1,180,160** |
| conv4 | DoubleConv(256→256) | 동일 = **1,180,160** |
| conv5 | DoubleConv(256→256) | 동일 = **1,180,160** |
| conv6 | DoubleConv(256→256) | 동일 = **1,180,160** |
| **인코더 소계** | | **5,908,052** |
| deconv1 | UpscaleDoubleConv(256→256) | 256×(256×9+1) × 2 = **1,180,160** |
| deconv2 | UpscaleDoubleConv(512→256) | 256×(512×9+1) + 256×(256×9+1) = **1,769,984** |
| deconv3 | UpscaleDoubleConv(512→256) | 동일 = **1,769,984** |
| deconv4 | UpscaleDoubleConv(512→256) | 동일 = **1,769,984** |
| deconv5 | UpscaleDoubleConv(512→256) | 동일 = **1,769,984** |
| deconv6 | UpscaleDoubleConv(512→179) | 179×(512×9+1) + 179×(179×9+1) = 824,237 + 288,122 = **1,112,359** |
| **디코더 소계** | | **9,372,455** |
| linear1 | Linear(256→256) | 256×256 + 256 = **65,792** |
| linear2 | Linear(256→130) | 256×130 + 130 = **33,410** |
| **합계** | | **15,379,709** |

```
총 파라미터: 15,379,709 ≈ 15.38M
메모리: 15,379,709 × 4 = 61.5 MB ≈ 0.057 GB
```

> U-Net skip connection 중간 feature 활성화: 입력 300×300 기준 각 스케일 feature를 유지해야 하므로 약 **+150 MB** 추가

---

### 4. BERT — `models/BERT_retriever/` 및 `planner/`

`SentenceTransformer('bert-base-nli-mean-tokens')` = **bert-base** 기반

```
Embedding:
  word_embeddings:     30,522 × 768 = 23,440,896
  position_embeddings:    512 × 768 =    393,216
  token_type_embeddings:    2 × 768 =      1,536
  LayerNorm:                2 × 768 =      1,536
  소계: 23,837,184

각 Transformer Layer (12개):
  Q,K,V projection: 3 × (768×768 + 768) = 1,771,776
  output projection: 768×768 + 768       =   590,592
  FFN fc1 (768→3072): 768×3072 + 3072   = 2,362,368
  FFN fc2 (3072→768): 3072×768 + 768    = 2,360,064
  LayerNorm × 2: 2×(2×768)              =     3,072
  레이어당: 7,087,872
  12 레이어: 85,054,464

Pooler dense: 768×768 + 768 = 590,592

총: 109,482,240 ≈ 109.5M
메모리: 109,482,240 × 4 = 437.9 MB ≈ 0.408 GB
```

> 단, BERT는 **추론 시에만** (subgoal 계획 단계) 사용. inference loop에서는 MMP_results JSON을 미리 로드해 사용하므로 실제 실행 중에는 GPU에 올릴 필요 없음.

---

### 5. Semantic Mapping 런타임 텐서 (`sem_mapping.py`)

파라미터 없음. 순전히 동적 버퍼.

**설정값 (코드 기반)**:
```
P=6,  num_sem_categories=615,  채널 수 = 616
screen_h = screen_w = 150 (alfred=1일 때)
vision_range = 100
max_height = int(360/5) = 72
min_height = int(-40/5) = -8
height_range = 72 - (-8) = 80
du_scale = 1
```

#### `init_grid` — 3D Voxel Buffer (가장 큰 텐서)
```
shape = [P, 1+num_sem_cat, vision_range, vision_range, height_range]
      = [6, 616, 100, 100, 80]

elements = 6 × 616 × 100 × 100 × 80 = 2,956,800,000
bytes    = 2,956,800,000 × 4        = 11,827,200,000
         ≈ 11.02 GB
```

#### `feat` — Feature 벡터 버퍼
```
shape = [P, 1+num_sem_cat, screen_h × screen_w]
      = [6, 616, 22,500]

elements = 6 × 616 × 22,500 = 83,160,000
bytes    = 83,160,000 × 4   = 332,640,000
         ≈ 0.310 GB
```

**Semantic Mapping 텐서 합계: 11.33 GB**

---

### 6. Full Map / Local Map (`main.py`)

```python
# nc = num_sem_categories + 4 = 615 + 4 = 619
# map_size = 1200cm // 5cm = 240 cells
# global_downscaling = 1  →  local = full = 240×240

full_map  = torch.zeros(6, 619, 240, 240)
local_map = torch.zeros(6, 619, 240, 240)
```

```
full_map:
  elements = 6 × 619 × 240 × 240 = 214,099,200
  bytes    = 214,099,200 × 4     = 856,396,800
           ≈ 0.798 GB

local_map: 동일 ≈ 0.798 GB

합계: 1,712,793,600 bytes ≈ 1.595 GB
```

---

## 전체 합계 (P=6 기준)

### 모델 가중치 메모리

| 모듈 | 파라미터 수 | 메모리 |
|------|-----------|--------|
| Mask R-CNN alfw_large | 46,228,430 | 0.172 GB |
| Mask R-CNN alfw_small | 46,454,474 | 0.173 GB |
| Mask R-CNN moca | 46,661,727 | 0.174 GB |
| UNetMulti (sem policy) | 1,291,177 | 0.005 GB |
| VALTS SimpleUNet (depth) | 15,379,709 | 0.057 GB |
| BERT (SentenceTransformer) | 109,482,240 | 0.408 GB |
| **가중치 합계** | **265,497,757 (265.5M)** | **0.989 GB** |

### 런타임 동적 텐서 메모리

| 텐서 | Shape | 메모리 |
|------|-------|--------|
| `init_grid` (voxel buffer) | [6, 616, 100, 100, 80] | **11.02 GB** |
| `feat` (feature buffer) | [6, 616, 22,500] | 0.310 GB |
| `full_map` | [6, 619, 240, 240] | 0.798 GB |
| `local_map` | [6, 619, 240, 240] | 0.798 GB |
| 중간 activation (MaskRCNN 3개) | — | ~0.9 GB |
| 중간 activation (depth U-Net) | — | ~0.15 GB |
| **런타임 텐서 합계** | | **13.98 GB** |

### 최종 합계

```
모델 가중치:           0.989 GB
런타임 동적 텐서:     13.976 GB
────────────────────────────────
총 GPU 메모리 추정:  ≈ 14.97 GB  (P=6, 단일 GPU 기준)

RTX A6000 (48 GB VRAM) 기준 여유분: ≈ 33 GB
→ 논문의 "48 GB 권장" 중 대부분의 여유는 AI2-THOR 렌더링 버퍼,
  CUDA context overhead, 그리고 더 많은 P를 허용하기 위한 것
```

---

## 핵심 분석 및 시사점

### 1. `init_grid`가 메모리를 압도한다

전체 GPU 메모리의 **약 73%**가 `init_grid` 하나에서 발생:

```
init_grid 메모리 ∝ P × (1 + 3 + 102×P) × vision_range² × height_range
```

`num_sem_categories = 3 + 102×P`이므로 P에 대해 **이차 스케일링**:
- P=1: 0.33 GB
- P=3: 2.58 GB
- P=6: **11.02 GB**
- P=11: **38.1 GB** (A6000 한계 접근)

→ **P가 늘어날수록 init_grid가 급증하는 것이 병렬화의 실질적 병목**

### 2. 채널 설계의 의도

`102×P`채널의 의미: 각 병렬 환경이 **최대 102개 물체 카테고리**를 독립적으로 추적.
모든 프로세스가 하나의 맵 모듈을 공유하므로, 채널 오프셋으로 프로세스를 구분.

### 3. BERT는 offline에서 사용

`MMP_results/*.json` 파일이 사전에 GPT-4+BERT로 생성되어 저장됨.
inference 시에는 JSON을 로드할 뿐, BERT가 GPU에 올라올 필요 없음 → **실제 inference 시 0.408 GB 절약**

### 4. 파라미터 대비 실효 계산량

가장 파라미터가 많은 BERT(109.5M)가 inference에서 사용 안 되고,
실제 realtime 추론의 핵심은 MaskRCNN(139.3M params, 0.519 GB) + 동적 텐서들.

---

## 파라미터 공식 참조

```
Conv2d(in, out, k):   out × (in × k × k + 1)
Linear(in, out):      in × out + out
Embedding(n, d):      n × d
BERT Layer:           7,087,872 (12heads, h=768, ffn=3072)
DoubleConv(cin,cout): cin×(cin×9+1) + cout×(cin×9+1)
UpscaleDoubleConv(cin,cout): cout×(cin×9+1) + cout×(cout×9+1)
```
