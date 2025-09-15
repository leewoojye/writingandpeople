# Genetic Team Scheduler 사용법

## 1. 가상환경 생성 및 활성화

### (1) Python 3.x 설치 필요

### (2) 터미널에서 아래 명령어 실행

```
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 또는
. .venv/bin/activate       # macOS/Linux (zsh/bash)
# Windows라면
.venv\Scripts\activate
```

## 2. 필수 패키지 설치

```
pip install -r requirements.txt
```

## 3. 입력 데이터 준비

- 입력 파일: `data/input/회원명단.csv`
- CSV 파일은 이름, 기수, 성별(남/여 또는 M/F) 컬럼이 있어야 합니다.

## 4. 소스코드 실행

```
python genetic/runner.py
```

## 5. 결과 확인

- 실행 후 `data/output/` 폴더에 `result_날짜_시간.csv` 형식의 결과 파일이 생성됩니다.

---

### 참고
- 인코딩 에러가 발생하면 입력 CSV 파일을 메모장 등으로 열어 "다른 이름으로 저장" → 인코딩을 `CP949` 또는 `EUC-KR`로 저장하세요.
- 가상환경을 종료하려면 `deactivate` 입력
