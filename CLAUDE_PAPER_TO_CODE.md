# FLARE: 논문 ↔ 코드 ↔ 메모리 연결 분석

**논문**: Multi-Modal Grounded Planning and Efficient Replanning For Learning Embodied Agents with a Few Examples (AAAI 2025)

---

## 논문이 제안하는 것: 두 가지 핵심 문제와 해법

### 왜 FLARE가 필요한가

기존 LLM-Planner (Song et al. 2023)의 문제:

```
지시: "Place a tray with a butter knife..."
LLM → 계획: (Navigate, StatueStep)
현실: StatueStep이 씬에 없음 → 에이전트가 무한 탐색 → 실패
```

원인이 두 가지다:
1. **계획 생성 시**: 환경 상태를 보지 않고 언어만으로 예시를 선택 → 엉뚱한 few-shot 예시 → 잘못된 계획
2. **계획 실행 시**: 잘못된 계획을 수정하려면 LLM을 또 호출 → 비용 과다

FLARE의 해법: **MMP**로 처음부터 환경-grounded 계획 생성, **EAR**로 LLM 없이 시각적으로 수정.

---

## Part 1: MMP (Multi-Modal Planner)

### 논문에서의 정의 (Section 3.1)

```
Sm = wl · (Sl / Σ sl,i) + we · (Se / Σ se,i)    ...(Eq. 1)
```

- `Sl`: BERT로 임베딩한 **언어 지시** 유사도
- `Se`: CLIP-ViT로 임베딩한 **초기 파노라마 뷰** 유사도
- 두 유사도를 정규화 후 합산 → top-k=9 훈련 예시 선택
- 선택된 예시를 few-shot prompt로 GPT-4에게 넘겨 subgoal 생성

subgoal 표현 (Eq. 2):
```
Sn = (An, On, Rn)
  An: 행동 ("pick", "clean", ...)
  On: 목표 물체 ("Apple")
  Rn: 수납 위치 ("Fridge")
```
이 triplet 표현이 (Song et al. 2023)보다 **토큰 25% 절감**.

### 코드 구현: `planner/`

#### Step 1 — 초기 파노라마 이미지 생성 (`planner/init_ego_generator.py`)

논문의 `Ce` (agent's surrounding views at command reception):
```python
# 씬 시작 시 에이전트 주변을 360도 촬영 → 파노라마 합성
# CLIP 인코더에 넣기 위한 입력 이미지
img_path = os.path.join(task_path, 'init_ego_panoramic.png')
```

#### Step 2 — 멀티모달 유사도 계산 (`planner/retriever.py`)

논문 Eq. 1을 그대로 구현:
```python
# 논문: Sl (언어 유사도)
model = SentenceTransformer('bert-base-nli-mean-tokens')
output_txt = model.encode(instruction)              # BERT 임베딩
text_similarities = np.dot(output_txt, train_text_dict.T)

# 논문: Se (환경 유사도)
model_clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
image_embeds = model_clip(**inputs).image_embeds    # CLIP 임베딩
img_similarities = np.dot(image_embeds, train_img_dict.T)

# 논문 Eq. 1: 정규화 후 합산 (wl=we=1)
text_sim_norm = text_similarities / np.linalg.norm(text_similarities)
img_sim_norm  = img_similarities  / np.linalg.norm(img_similarities)
combined = IMG_WEIGHT * img_sim_norm + TXT_WEIGHT * text_sim_norm

# top-k=9 선택
similar_keys = find_most_similar_keys(combined, train_text_dict, k=9)
```

#### Step 3 — GPT-4로 subgoal 생성 (`planner/generate_plans.py`)

```python
# 9개 few-shot 예시 + 현재 지시 → GPT-4 API 호출
retrived = json.load(f'few_examples_from_song/few-song-{sp}_retrieved_keys...')
few_examples = json.load('few_examples_from_song/few_examples.json')

# 논문의 triplet (An, On, Rn) 형식으로 subgoal 추출
result[instruction]['triplet'] = []   # (action, object, receptacle)
```

#### Step 4 — 결과 저장 (`MMP_results/*.json`)

```
MMP_results/
  valid_seen.json      # {instruction: {task_type, object_target, parent_target, ...}}
  valid_unseen.json
  tests_seen.json
  tests_unseen.json
```

**Inference 시에는 GPT-4/BERT/CLIP을 호출하지 않는다.**
`ALFRED_task_helper.py`의 `read_test_dict()`가 JSON을 로드하여 사용.

#### Step 5 — 실행 시 subgoal 파싱 (`models/instructions_processed_LP/ALFRED_task_helper.py`)

```python
def get_list_of_highlevel_actions(traj_data, ...):
    # MMP 결과 JSON에서 subgoal 리스트 반환
    # → [("Find", "Apple"), ("PickUp", "Apple"), ("Find", "Fridge"), ("PutIn", "Apple")]
    # caution_pointers: 조심해야 할 subgoal 인덱스
    # convert_pointers: 카테고리 변환 필요 지점 (Knife↔ButterKnife)
```

### MMP 관련 메모리 footprint

| 컴포넌트 | 메모리 | 비고 |
|---------|--------|------|
| BERT (SentenceTransformer) | 0.408 GB | **Offline 전처리만** 사용. Inference 시 GPU에 불필요 |
| CLIP-ViT (clip-vit-base-patch32) | ~0.35 GB | **Offline 전처리만** 사용 |
| GPT-4 | API 호출 (로컬 메모리 없음) | Offline 계획 생성 시만 |
| `MMP_results/*.json` | ~수 MB | RAM에 dict로 로드 |
| **Inference 시 실제 부담** | **≈ 0 GB** | JSON 로드만 |

**핵심**: MMP는 inference loop에서 메모리를 거의 쓰지 않는다. 무거운 모델은 모두 offline에서 돌리고 결과를 캐싱한다.

---

## Part 2: EAR (Environment Adaptive Replanning)

### 논문에서의 정의 (Section 3.2, Algorithm 1)

```
V* = argmax_Vi SC(Enc(Ok), Enc(Vi))    ...(Eq. 3)
```

- `Ok`: 현재 subgoal의 목표 물체 (예: "StatueStep")
- `Vi`: 에이전트가 지금까지 발견한 물체들의 집합 V
- `Enc(·)`: 언어 인코더 (BERT)
- `SC(·,·)`: 코사인 유사도
- 불확실도 `u`가 임계값 `τ`를 초과하면 EAR 발동 → `Ok`를 `V*`로 교체

Algorithm 1의 전체 루프:
```
P ← MMP(I, C0)              초기 계획 생성
S ← SemanticMapping(C0)     시맨틱 맵 초기화

while k < len(P):
    Ct ← Execute(at)
    S ← SemanticMapping(Ct)  매 스텝 맵 갱신
    V.add(ObjectDetector(Ct)) 발견 물체 누적

    if Ok not in O:
        u += 1
        if u > τ:
            Ok ← EAR(Ok, V)  ← BERT 코사인 유사도로 교체
    elif Complete(Pk):
        k += 1               다음 subgoal로
```

### 코드 구현: `agents/sem_exp_thor.py` + `models/BERT_retriever/`

#### EAR 트리거 조건: `confuse` 카운터

```python
# sem_exp_thor.py
CONFUSE_THRESHOLD = 200       # 논문의 불확실도 임계값 τ

self.confuse = 0              # 논문의 u (불확실도 누적값)
self.hallu2grounded = dict()  # 교체 매핑 캐시 (hallucinated → grounded)
```

에이전트가 목표 물체를 못 찾고 같은 자리를 200스텝 이상 배회하면 EAR 발동.

#### EAR 실행: BERT 코사인 유사도 (`models/BERT_retriever/BERT_retriever.py`)

논문 Eq. 3을 그대로 구현:
```python
def retrieve(candidate, objects_in_scene, ispickupable):
    # candidate = Ok (현재 못 찾는 물체 이름)
    # objects_in_scene = V (지금까지 발견한 물체들)

    model = SentenceTransformer('bert-base-nli-mean-tokens')  # 논문의 Enc(·)

    # 발견된 물체들의 임베딩 DB 로드
    if ispickupable:
        train_dict = pickle.load('models/BERT_retriever/pickupable_NoLamp_emb.p')
    else:
        train_dict = pickle.load('models/BERT_retriever/recep_emb.p')

    # objects_in_scene 중 train_dict에 있는 것만 후보로
    emb_pool = {k: train_dict[k] for k in objects_in_scene if k in train_dict}

    # 논문 Eq. 3: V* = argmax SC(Enc(Ok), Enc(Vi))
    output = model.encode([candidate]).squeeze()                # Enc(Ok)
    similar_keys = find_most_similar_keys(output, emb_pool, k=1)  # SC 최대화
    return similar_keys[0]  # V*
```

임베딩 DB 사전 구축 (`models/BERT_retriever/BERT_converter.py`):
```python
# ALFRED 모든 물체명을 BERT 임베딩으로 변환해서 저장
model = SentenceTransformer('bert-base-nli-mean-tokens')
for obj in alfred_objects:
    pickupable_dict[obj] = model.encode(str(obj))
pickle.dump(pickupable_dict, open('pickupable_emb.p', 'wb'))
```

#### V 집합 유지: `seg.objects_in_scene`

```python
# segmentation_helper.py
self.objects_in_scene = set()   # 논문의 V (발견 물체 누적 집합)

# 매 스텝 ObjectDetector 결과로 갱신
# → Algorithm 1의 V.add(ObjectDetector(Ct))
```

### EAR 관련 메모리 footprint

| 컴포넌트 | 메모리 | 비고 |
|---------|--------|------|
| BERT (EAR용) | 0.408 GB | **MMP와 다른 BERT 인스턴스** — EAR은 Inference 시 실시간 사용 |
| 임베딩 DB (`*.p` 파일) | ~수십 MB RAM | pickupable (~74개 물체), recep (32개) 임베딩 pkl |
| `objects_in_scene` set | 무시 가능 | 문자열 집합, KB 수준 |

**EAR의 메모리 특성**: BERT가 inference 중 실시간으로 필요. EAR은 MMP와 달리 offline 캐싱이 불가능 — 에이전트가 어떤 물체를 발견했는지는 실행 중에만 알 수 있기 때문이다.

---

## Part 3: Action Policy — 논문 Section 3.3

논문에서 명시적으로:
> "We adopt the deterministic approach (Sethian 1996) for effective path planning"
> (FMM: Fast Marching Method)

이것이 **Semantic Mapping + FMM Planner**의 역할이다.

### 시맨틱 맵 (`sem_mapping.py`) ↔ 논문 Algorithm 1의 S

```python
# Algorithm 1:
#   S ← SemanticMapping(Ct)      ← 바로 이 모듈
#   at ← ActionPolicy(Pk, S)     ← FMMPlanner가 S를 입력으로 받음

# forward() 출력:
#   fp_map_pred  → S의 장애물 채널
#   map_pred     → S 전체 (누적 시맨틱 맵)
#   current_poses → 에이전트 현재 위치
```

### FMM Planner (`envs/utils/fmm_planner.py`) ↔ 논문의 ActionPolicy

```python
# S (시맨틱 맵)에서 목표 물체 위치 추출
# → FMMPlanner(traversible)  # 장애물 맵을 traversable 맵으로
# → planner.set_goal(goal)   # Pk의 On 위치를 목표로
# → skfmm.distance(...)      # 전체 맵에 거리장 계산
# → 에이전트 위치에서 gradient 방향으로 한 스텝
```

### 이 과정에서의 메모리: `init_grid`의 존재 이유

`init_grid = torch.zeros(P, 616, 100, 100, 80)` → **11.02 GB**

이 텐서가 왜 이렇게 큰가를 논문의 목적과 연결하면:

```
논문 목적: 에이전트가 방을 탐색하면서 시맨틱 맵 S를 누적
           → 어디에 어떤 물체가 있는지, 어디가 탐색됐는지 기억

코드의 init_grid 역할:
  - 3D voxel grid: XY(평면) × Z(높이)
  - 채널 616개 = 1(obstacle) + 615(카테고리 × 프로세스)
  - 매 프레임 카메라 → 포인트 클라우드 → 이 voxel에 splatting
  - 높이 방향 collapse → 2D 시맨틱 맵 S 생성

채널이 102×P인 이유:
  P개 환경이 각각 다른 물체를 목표로 함
  → 각 환경의 시맨틱 채널을 채널 오프셋으로 분리
  → 배치 처리 가능 (GPU 병렬화)
```

---

## 논문 Figure 2 ↔ 코드 파일 매핑

```
Figure 2에 나온 구조:
┌─────────────────────────────────────────────────────────────┐
│  초기 파노라마 뷰 + 언어 지시                                │
│           │                                                  │
│    ┌──────▼──────┐                                          │
│    │     MMP     │ planner/retriever.py                     │
│    │  BERT+CLIP  │ planner/generate_plans.py                │
│    │  → GPT-4   │ → MMP_results/*.json (캐시)              │
│    └──────┬──────┘                                          │
│           │ Subgoal sequence: [(An, On, Rn), ...]           │
│           ▼                                                  │
│    ┌─────────────────────────────────────────────────┐      │
│    │              Execution Loop (main.py)            │      │
│    │                                                  │      │
│    │  RGB-D 관찰                                      │      │
│    │    │                                             │      │
│    │    ├→ [Object Detector]  segmentation_helper.py │      │
│    │    │   Mask R-CNN × 3                           │      │
│    │    │   → V (발견 물체 집합)                     │      │
│    │    │                                             │      │
│    │    ├→ [Semantic Mapping]  sem_mapping.py         │      │
│    │    │   Depth → PointCloud → Voxel → 2D Map S    │      │
│    │    │                                             │      │
│    │    └→ [FMM Planner]  fmm_planner.py             │      │
│    │        S → 경로 → 행동 at                        │      │
│    │                                                  │      │
│    │  if u > τ: ──→ [EAR]  BERT_retriever.py         │      │
│    │               Ok → V* (BERT 코사인 유사도)       │      │
│    └─────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 메모리 레이아웃과 논문 설계 의도의 연결

### 왜 메모리가 이렇게 배분되었는가

| 논문 컴포넌트 | 메모리 | 설계 의도 |
|-------------|--------|----------|
| **MMP** (BERT+CLIP+GPT-4) | **≈ 0 GB** (inference 시) | Offline 전처리 후 JSON 캐싱. "computationally efficient replanning without LLM" 목표 |
| **EAR** (BERT) | **0.408 GB** | Inference 중 실시간 필요. 하지만 LLM 전체 재호출보다 훨씬 저렴 |
| **Semantic Mapping** (`init_grid`) | **11.02 GB** | P개 환경의 3D 시맨틱 공간을 실시간 유지. 논문의 "환경 상태를 반영하는 S" |
| **Mask R-CNN × 3** | **0.519 GB** | 논문의 ObjectDetector — V 집합 구축 및 S의 시맨틱 채널 공급 |
| **Full/Local Map** | **1.595 GB** | S (누적 시맨틱 맵) 자체. FMM Planner의 입력 |

### 논문이 주장하는 "효율성"의 실제 의미

논문: *"our FLARE does not use LLMs for replanning to improve computational efficiency"*

코드에서 이것이 어떻게 실현되는가:

```
LLM-Planner (기존):
  실행 중 막히면 → LLM API 재호출 → 전체 subgoal 재생성
  → 매 실패마다 수초의 API 대기 + 비용

FLARE EAR:
  실행 중 막히면 → BERT 임베딩 로컬 조회 → 하나의 subgoal만 교체
  → 밀리초 수준, 추가 API 비용 없음

  단, BERT는 상시 GPU 메모리에 올려야 함 → 0.408 GB
  대신 GPT-4 API 비용 0
```

### `init_grid` 크기가 11 GB인 것은 논문 설계의 직접적 결과

논문의 Table 1에서 P=6으로 48개 인스턴스를 동시 실행 (`leaderboard_all.sh`):
```bash
# leaderboard_all.sh: 8 GPU × 6 프로세스 = 48 병렬 실행
# 각 GPU당 P=6
# num_sem_categories = 3 + 102×6 = 615
# init_grid = [6, 616, 100, 100, 80] = 11.02 GB/GPU
```

이것이 RTX A6000 (48 GB)이 필요한 진짜 이유:
- 모델 가중치: ~1 GB
- `init_grid`: ~11 GB
- Full/Local Map: ~1.6 GB
- 나머지: ~2 GB
- **합계 ≈ 15 GB** → A6000의 48 GB에서 약 33 GB 여유

여유 33 GB는 AI2-THOR 렌더러(3D 씬 렌더링), CUDA context, 그리고 **더 많은 P를 수용하기 위한 버퍼**.

### `102 × P` 채널 설계의 논문적 의미

논문은 **few-shot (0.5% = 100개 예시)** 설정에서 ALFRED의 7가지 task type, 수백 개의 물체를 다룬다. 각 episode마다 에이전트는 다른 물체를 찾아야 하고, P개 환경이 동시에 다른 목표를 갖는다.

```
102개 카테고리/프로세스:
  - ALFRED 주요 물체 (~74 pickupable)
  - receptacle (~32)
  - 합계 ≈ 102

채널 오프셋:
  env 0의 물체 맵 → 채널 4   ~ 105
  env 1의 물체 맵 → 채널 106 ~ 207
  ...
  env 5의 물체 맵 → 채널 514 ~ 615
```

이 설계 덕분에 **P개 환경의 맵핑을 단일 배치 연산으로 처리** → GPU 활용률 극대화.

---

## Ablation Study (논문 Table 3) ↔ 코드 옵션

| 실험 | MMP | EAR | SR (seen/unseen) | 코드 설정 |
|------|-----|-----|-----------------|----------|
| (a) 전체 FLARE | ✓ | ✓ | 32.55 / 31.79 | 기본 실행 |
| (b) EAR만 | ✗ | ✓ | 30.20 / 30.35 | `--no_caution_pointers` 류 |
| (c) MMP만 | ✓ | ✗ | 30.79 / 30.28 | `CONFUSE_THRESHOLD = ∞` |
| (d) 없음 | ✗ | ✗ | 28.05 / 28.58 | 둘 다 비활성 |

- MMP 제거 효과: -1.76%p SR (unimodal 예시 선택의 한계)
- EAR 제거 효과: -1.51%p SR (잘못된 subgoal을 고치지 못함)
- 둘 다 있을 때 상호보완: 4.5%p 향상

---

## 요약: 논문 아이디어 → 코드 → 메모리의 흐름

```
논문 문제 인식                코드 해법                메모리 영향
─────────────────────────────────────────────────────────────────
"언어만으로 예시 선택 →       planner/retriever.py     ≈ 0 GB (offline)
 환경 무시한 계획"            BERT + CLIP 멀티모달
      ↓ MMP                   → MMP_results/*.json     수 MB (RAM)

"잘못된 subgoal →             BERT_retriever.py        0.408 GB (GPU,
 LLM 재호출 비용"             BERT 코사인 유사도         상시 유지)
      ↓ EAR                   로컬에서 O(n) 검색

"환경 상태 추적 →             sem_mapping.py           11.02 GB ←
 어디에 뭐가 있는지"          init_grid voxel buffer    전체의 73%
      ↓ S (시맨틱 맵)          full_map / local_map     1.595 GB

"물체 감지 →                  segmentation_helper.py   0.519 GB
 V 집합 구축"                  Mask R-CNN × 3           (3개 모델)

"경로 계획"                   fmm_planner.py           무시 가능
      ↓ ActionPolicy          skfmm 라이브러리 (CPU)    (NumPy)
─────────────────────────────────────────────────────────────────
전체 GPU 메모리 ≈ 15 GB (P=6, RTX A6000 48 GB 사용)
```
