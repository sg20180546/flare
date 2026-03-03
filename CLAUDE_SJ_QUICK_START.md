# FLARE H100 Quick Start (Python 3.8 + CUDA 12.2)

## 환경 요약
- GPU: NVIDIA H100 80GB HBM3
- CUDA: 12.2
- Python: 3.8
- OS: Ubuntu (headless server)

---

## 1. conda 환경 생성

```bash
conda create -n flare38 python=3.8 -y
conda activate flare38
```

---

## 2. PyTorch 설치 (CUDA 12.1 wheel이 12.2와 호환)

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. requirements.txt 수정 후 설치

`requirements.txt`에서 아래 두 줄 수정:
- `dataclasses==0.8` → 삭제 (Python 3.8 내장)
- `scikit-fmm==2019.1.30` → `scikit-fmm` (버전 고정 해제)

```bash
pip install numpy==1.23.5
pip install -r requirements.txt
```

---

## 4. 추가 패키지 설치 (requirements.txt 누락)

```bash
# scikit-image 버전 업그레이드 (0.15.0은 numpy API 불일치)
pip install scikit-image==0.19.3

# setuptools 다운그레이드 (detectron2 빌드용)
pip install setuptools==59.5.0

# detectron2 (C++ 컴파일 포함, 시간 걸림)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# sentence-transformers (BERT retriever용)
pip install sentence-transformers
```

---

## 5. Pretrained 모델 다운로드

```bash
cd ~/flare

# FILM 모델 (Google Drive)
pip install gdown
gdown "https://drive.google.com/uc?id=1mkypSblrc0U3k3kGcuPzVOaY1Rt9Lqpa" -O Pretrained_Models_FILM.zip
unzip Pretrained_Models_FILM.zip
mv Pretrained_Models_FILM/maskrcnn_alfworld models/segmentation/maskrcnn_alfworld
mv Pretrained_Models_FILM/depth_models models/depth/depth_models
mv Pretrained_Models_FILM/new_best_model.pt models/semantic_policy/best_model_multi.pt

# MOCA MaskRCNN (S3 직접 다운)
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt
```

---

## 6. ALFRED 데이터 준비

```bash
# alfred repo 클론 (flare와 별도 경로)
cd ~
git clone https://github.com/askforalfred/alfred.git
cd alfred
export ALFRED_ROOT=$(pwd)

# 전처리 (5~15분, 마지막에 에러 나도 정상)
python models/train/train_seq2seq.py --data data/json_2.1.0/ --splits data/splits/oct21.json --preprocess

# flare로 돌아와서 심볼릭 링크 연결
cd ~/flare
rm alfred_data_all/json_2.1.0
rm alfred_data_all/splits
ln -s /home/elicer/alfred/data/json_2.1.0 alfred_data_all/json_2.1.0
ln -s /home/elicer/alfred/data/splits alfred_data_all/splits
```

---

## 7. X서버 설정

```bash
# pciutils 설치 (lspci 필요)
sudo apt-get install -y pciutils xorg

# Xwrapper 권한 설정
sudo bash -c 'echo "allowed_users=anybody" > /etc/X11/Xwrapper.config'

# tmux에서 X서버 실행
tmux new -s xserver
conda activate flare38
python alfred_utils/scripts/startx.py 0
# Ctrl+B → D 로 tmux 나오기

export DISPLAY=:0
```

> X서버가 이미 떠 있으면 (`Server is already active for display 0`) 그냥 `export DISPLAY=:0` 만 하면 됨

---

## 8. eval.sh 수정

`eval.sh`의 `--x_display 1` → `--x_display 0` 으로 변경:

```bash
# eval.sh line 26
--x_display 0   \
```

---

## 9. 실행

### 테스트 (에피소드 5개)
```bash
cd ~/flare
conda activate flare38
export DISPLAY=:0
CUDA_VISIBLE_DEVICES=0 bash eval.sh tests_unseen 0 5 flare
```

### 전체 실행 (GPU 1장 기준, tests_unseen 전체)
```bash
CUDA_VISIBLE_DEVICES=0 bash eval.sh tests_unseen 0 1529 flare
```

### 원래 leaderboard_all.sh는 8 GPU 기준 → GPU 1장이면 위처럼 직접 호출

---

## 환경 검증 커맨드

```bash
cd ~/flare
conda activate flare38

# 심볼릭 링크 확인
ls -la alfred_data_all/

# 모델 파일 확인
ls models/segmentation/maskrcnn_alfworld/   # objects_lr5e-3_005.pth, receps_lr5e-3_003.pth
ls models/depth/depth_models/               # valts, valts0
ls models/semantic_policy/best_model_multi.pt
ls weight_maskrcnn.pt

# PyTorch + CUDA 확인
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 기대값: 2.1.0+cu121 / True / NVIDIA H100 80GB HBM3
```
