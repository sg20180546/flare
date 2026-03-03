# FLARE 프로젝트 전체 구조

**논문**: [Multi-Modal Grounded Planning and Efficient Replanning For Learning Embodied Agents with A Few Examples](https://arxiv.org/abs/2412.17288) (AAAI 2025)
**태스크**: ALFRED benchmark — 가상 실내 환경(AI2-THOR)에서 자연어 지시를 받아 물체를 조작하는 embodied agent
**기반**: [FILM](https://github.com/soyeonm/FILM), [CAPEAM](https://github.com/snumprlab/capeam)

---

## 시스템 전체 그림

```
자연어 지시 ("사과를 냉장고에 넣어줘")
        │
        ▼
[ 1. MMP: 멀티모달 플래너 ]   ← BERT + CLIP으로 유사 예시 검색 → GPT-4로 subgoal 생성
   "Find Apple → PickUp Apple → Find Fridge → PutIn Apple"
        │
        ▼
[ 2. 실행 루프 (main.py + agents/sem_exp_thor.py) ]
   각 subgoal 처리:
   ┌─────────────────────────────────────────────┐
   │  RGB + Depth 관찰                            │
   │       │                                      │
   │  [인식] Mask R-CNN → 물체 segmentation       │
   │       │                                      │
   │  [맵핑] sem_mapping.py → 2D 시맨틱 맵 갱신   │
   │       │                                      │
   │  [목표] UNet semantic policy → 탐색 목표 결정 │
   │       │                                      │
   │  [경로] FMM Planner → 장애물 회피 경로        │
   │       │                                      │
   │  [행동] 이동 / 조작 (pick, put, open, ...)    │
   └─────────────────────────────────────────────┘
        │
        ▼
   성공 or 실패 → 다음 subgoal
```

---

## 디렉토리별 역할

---

### `planner/` — MMP (Multi-Modal Planner): FLARE의 핵심 기여

FILM과 다른 FLARE만의 차별점. **few-shot으로 subgoal 시퀀스를 생성**한다.

#### `planner/retriever.py` — 멀티모달 유사 예시 검색

```python
# BERT로 언어 임베딩 + CLIP으로 초기 관찰 이미지 임베딩
text_similarities  = BERT(instruction) vs train_text_dict
img_similarities   = CLIP(panoramic_image) vs train_img_dict

# 두 유사도를 합산해 가장 비슷한 훈련 예시 9개 검색
combined = IMG_WEIGHT * img_sim + TXT_WEIGHT * text_sim
```

- **언어만 쓰면**: "사과를 냉장고에" → 비슷한 지시 검색
- **이미지도 쓰면**: 방의 초기 모습(어떤 물체가 있는지)까지 반영해서 더 정확한 예시 선택
- 결과는 `MMP_results/*.json`에 사전 저장됨 (inference 시 로드)

#### `planner/generate_plans.py` — GPT-4로 subgoal 생성

```python
# 검색된 9개 훈련 예시를 few-shot prompt로 구성
# GPT-4에게 "이 지시의 subgoal 시퀀스는?" 질문
retrived = 유사_예시_9개
result = GPT4(few_shot_prompt + 현재_지시)
# → ["Find Apple", "PickUp Apple", "Find Fridge", "PutIn Apple"]
```

- `planner/few_examples_from_song/` 안의 훈련 예시 임베딩이 DB
- 결과가 `MMP_results/*.json`에 저장됨

#### `planner/postprocess.py` — 생성 결과 후처리
#### `planner/init_ego_generator.py` — 초기 파노라마 이미지 생성 (CLIP 입력용)

---

### `main.py` — 메인 실행 루프

전체 에피소드 루프를 관리한다.

```python
# 주요 흐름
args = get_args()
envs = make_vec_envs(...)          # 병렬 환경 생성 (num_processes개)
sem_map_module = Semantic_Mapping(args)
sem_map_model  = UNetMulti(...)    # semantic policy (탐색 목표 결정)

while 에피소드 남음:
    obs, info, actions_dict = env.load_initial_scene()  # 씬 로드 + MMP 결과 읽기

    for subgoal in actions_dict['list_of_actions']:
        while subgoal 미완료:
            # 맵 갱신
            fp_map, map_pred, pose_pred, current_poses = sem_map_module(obs, pose_obs, maps_last, poses_last)

            # 탐색 목표 결정 (semantic policy or 랜덤)
            if use_sem_policy:
                global_goals = sem_map_model(map_pred)  # UNet이 어디 갈지 결정
            else:
                global_goals = random_unexplored()

            # FMM으로 경로 계획 → 행동 실행
            action = fmm_planner.get_next_action(global_goals)
            obs, reward, done, info = env.step(action)
```

**`num_sem_categories = 1 + 1 + 1 + 102 * num_processes`**
→ obstacle + explored + (인스턴스별 물체) × 프로세스 수 (각 프로세스가 다른 목표 물체)

---

### `agents/sem_exp_thor.py` — 에이전트 행동 로직

`main.py`의 루프에서 실제 행동을 결정하는 클래스.

#### 초기화 시:
- MMP 결과(`MMP_results/*.json`) 로드 → subgoal 리스트 파악
- Segmentation 모델 초기화 (`SemgnetationHelper`)
- `list_of_actions`: `[("PickupObject", "Apple"), ("PutObject", "Fridge"), ...]` 형태

#### 각 subgoal 처리:
```
Navigate → Interact (pick/put/open/slice/...) → Navigate 반복
```

- **이동**: `FMMPlanner`가 계산한 waypoint로 MoveAhead / RotateLeft / RotateRight
- **조작**: `va_interact_new()` — AI2-THOR API 호출
- **실패 처리**: `fails_cur` 카운트, side-step, look-around 등 복구 행동

#### 특이한 처리들:
- `transfer_cat`: ButterKnife ↔ Knife 혼용 처리
- `confuse`: 같은 자리를 반복 탐색하면 hallucination으로 판단 → BERT로 다른 물체 후보 검색
- `last_three_sidesteps`: 반복 side-step 감지
- `camera_horizon`: look up/down 각도 추적 및 복원

---

### `models/` — 인식 + 정책 모델들

#### `models/segmentation/` — 물체 인식 (Mask R-CNN)

```python
# SemgnetationHelper가 3개 모델 동시 운용:
sem_seg_model_alfw_large  # 수납장 등 큰 물체 (receptacle)
sem_seg_model_alfw_small  # 사과, 컵 등 작은 물체 (pickupable)
sem_seg_model_moca        # MOCA pretrained MaskRCNN (119 classes)
```

- 입력: RGB 이미지
- 출력: 물체별 instance mask + class label
- `alfworld_constants.py`: ALFRED 물체 카테고리 목록 정의

#### `models/semantic_policy/sem_map_model.py` — 탐색 목표 결정 (UNet)

```python
class UNetMulti(nn.Module):
    # 입력: 누적 시맨틱 맵 (num_sem_categories 채널, 240×240)
    # 출력: 탐색 목표 위치 (heat map)

    main = Conv→ReLU→MaxPool × 4   # 맵 특징 추출
    goal_emb = Embedding(73, 256)  # 목표 물체 임베딩
    linear → softmax               # 어느 맵 셀로 갈지
```

- "아직 탐색 안 된 곳" + "목표 물체가 있을 법한 곳" 을 종합해 목표 결정
- `--use_sem_policy` 없으면 완전 random 탐색

#### `models/BERT_retriever/` — 시각적 환경에서 물체 이름 매칭

```python
def retrieve(candidate, objects_in_scene, ispickupable):
    # candidate: 지시문에 나온 물체 이름 (e.g. "the red cup")
    # objects_in_scene: 현재 씬에서 발견된 물체들
    # BERT 임베딩 코사인 유사도로 가장 가까운 실제 물체 반환
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    output = model.encode(candidate)
    → 코사인 유사도 최대인 물체 반환
```

- 지시에서 "knife"라고 했는데 씬에 "ButterKnife"만 있으면 → 매칭
- `BERT_converter.py`: 모든 ALFRED 물체명을 BERT 임베딩으로 변환해서 `.p` 파일에 저장 (전처리)

#### `models/instructions_processed_LP/ALFRED_task_helper.py` — 지시문 파싱

```python
def get_list_of_highlevel_actions(traj_data, ...):
    # MMP가 생성한 subgoal JSON을 읽어서:
    # [("Find", "Apple"), ("PickUp", "Apple"), ("Find", "Fridge"), ("PutIn", "Apple")] 반환
    # caution_pointers: 특히 주의할 subgoal 인덱스
    # convert_pointers: 물체 카테고리 변환이 필요한 지점
```

#### `models/depth/` — Depth 추정 모델

```python
# VALTS depth 모델 (학습된 monocular depth estimation)
# --learned_depth 플래그로 활성화
# 기본은 AI2-THOR 시뮬레이터의 ground truth depth 사용
```

---

### `sem_mapping.py` / `models/sem_mapping.py` — 2D 시맨틱 맵 구축

(루트의 `sem_mapping.py`와 `models/sem_mapping.py`는 동일한 파일)

RGB-D + 세그멘테이션 결과 → 2D 누적 맵으로 변환.
자세한 구현은 아래 섹션 참조.

---

### `envs/utils/` — 맵핑 수학 유틸

| 파일 | 역할 |
|---|---|
| `depth_utils.py` | Depth → 3D Point Cloud, voxel splatting, 좌표 변환 |
| `fmm_planner.py` | Fast Marching Method 경로 계획 (skfmm 라이브러리 사용) |
| `rotation_utils.py` | 회전 행렬 생성 (Rodrigues 공식) |
| `pose.py` | pose 변환 유틸 |
| `map_builder.py` | 맵 구축 보조 함수 |
| `vector_env.py` | 멀티프로세스 환경 래퍼 |

#### `fmm_planner.py` — Fast Marching Method 경로 계획
```python
# 장애물 맵을 마스킹 → 목표 지점으로의 거리장(distance field) 계산
traversible_ma = ma.masked_values(traversible * 1, 0)
dd = skfmm.distance(traversible_ma, dx=scale)
# 에이전트 위치에서 gradient 방향으로 한 스텝씩 이동
```

---

### `alfred_utils/` — AI2-THOR 환경 인터페이스

ALFRED 공식 코드를 래핑한 유틸리티들. 거의 수정 없이 사용.

| 파일 | 역할 |
|---|---|
| `env/thor_env.py` | AI2-THOR 환경 기본 클래스 |
| `env/thor_env_code.py` | FLARE가 상속해서 사용하는 확장 클래스 |
| `gen/constants.py` | ALFRED 물체 목록, 맵 크기 등 상수 |
| `gen/ff_planner/` | Fast-Forward PDDL 플래너 (C 코드, 훈련 데이터 생성용) |
| `models/eval/` | 평가 스크립트 |

---

### `utils/` — 기타 유틸

| 파일 | 역할 |
|---|---|
| `control_helper.py` | 에이전트 이동 방향 계산, 충돌 감지, action sequence 기록 |
| `model.py` | `get_grid` (Spatial Transformer용 변환행렬), `ChannelPool`, `NNBase` |
| `distributions.py` | RL 정책용 Categorical/DiagGaussian 분포 |
| `storage.py` | RL rollout 저장 버퍼 |
| `leaderboard_script.py` | ALFRED 리더보드 제출 형식 변환 |

---

### `MMP_results/` — 사전 생성된 subgoal 시퀀스

```
MMP_results/
  tests_seen.json          # test-seen split, MMP subgoals (appended)
  tests_unseen.json        # test-unseen split
  valid_seen.json          # valid-seen split
  valid_unseen.json        # valid-unseen split
  tests_*_no_append.json   # high-level description 미포함 버전
```

런타임에 GPT-4를 호출하지 않고 여기서 로드. `ALFRED_task_helper.py`의 `read_test_dict()`가 담당.

---

### `arguments.py` — 전체 설정

주요 설정값:

| 플래그 | 의미 |
|---|---|
| `--use_sem_policy` | UNet semantic policy 사용 (없으면 random 탐색) |
| `--use_sem_seg` | Mask R-CNN 세그멘테이션 사용 |
| `--learned_depth` | 학습된 depth 모델 사용 (기본은 GT depth) |
| `--appended` | MMP subgoal에 high-level description 포함 여부 |
| `--eval_split` | 평가 split 선택 |
| `--num_processes` | 병렬 환경 수 (GPU 메모리에 따라 자동 설정) |

---

## 실행 흐름 요약

```
bash leaderboard_all.sh
    │
    └─ python main.py --eval_split tests_unseen --appended ...
            │
            ├─ MMP_results/*.json 로드 (GPT-4가 사전 생성한 subgoal)
            │
            ├─ num_processes개 AI2-THOR 환경 병렬 실행
            │
            └─ 에피소드 루프:
                  subgoal = ["Find Apple", "PickUp Apple", "PutIn Fridge"]

                  for each subgoal:
                      while 미완료:
                          obs → [Seg] → [SemMap] → [FMM] → action → env.step()
```

---

## `sem_mapping.py` 구현 상세

### 맵 채널 구조

| 채널 | 의미 |
|---|---|
| 0 | 장애물 (Obstacle) |
| 1 | 탐색됨 (Explored) |
| 2, 3 | 외부에서 설정 (에이전트 위치 등) |
| 4 ~ 4+N | 시맨틱 카테고리별 물체 위치 |

### `forward()` 파이프라인

```
depth 이미지
    │
    ▼
get_point_cloud_from_z_t()     # 픽셀 → 3D XYZ (핀홀 카메라 역투영)
    │  X = (u - cx) * depth / f
    │  Z = (v - cy) * depth / f
    ▼
transform_camera_view_t_multiple()   # 카메라 틸트 보정 + 높이 오프셋
    │  block_diag으로 num_processes 동시 처리
    ▼
transform_pose_t()             # 에이전트 중심 좌표로 정렬
    ▼
좌표 [-1, 1] 정규화
    ▼
splat_feat_nd()                # 3D voxel에 feature 투영 (bilinear splatting)
    │  scatter_add_로 인접 voxel에 가중치 분산
    ▼
높이 슬라이싱 → 2D 투영
    │  [5cm ~ 카메라높이+50cm] → 장애물 맵
    │  [전체 높이] → 탐색 맵
    ▼
Spatial Transformer (grid_sample)   # 에이전트 시점 → 글로벌 좌표계
    ▼
torch.max(maps_last, current)  # 이전 맵과 max merge → 누적
```
