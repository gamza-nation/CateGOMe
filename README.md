# CateGOMe (카테고미) - 가계동향조사 항목 자동분류 AI

<div align="center">
  <img src="assets/CateGOMe_kor.png" width="420" alt="CateGOMe Logo">
  <p>통계청 가계동향조사를 위한 지능형 항목코드 자동분류 시스템</p>
</div>

## 📌 프로젝트 개요

CateGOMe는 통계청 직원 및 조사원들이 가계부 항목을 표준 분류코드로 매핑하는 복잡한 작업을 자동화하는 AI 어시스턴트입니다. 가계부 이미지를 업로드하면 OCR을 통해 품목을 추출하고, RAG(Retrieval-Augmented Generation) 기반 AI가 각 항목에 대한 적절한 통계 분류코드를 자동으로 할당합니다.

### 🎯 주요 해결 과제

1. **복잡한 분류 체계**: 동일 품목도 세부 특성에 따라 다른 코드 부여 (예: 간고등어 vs 통조림고등어 vs 생고등어)
2. **신규 품목 대응**: ChatGPT 구독료, 전기차 충전비 등 기존 분류집에 없는 신규 항목 처리
3. **모호한 표현 해석**: "초코뿅망치" → "뿅망치 구슬초코" 같은 일상 표현 이해
4. **우선순위 규칙 적용**: 조사사례집 > 항목분류집 순서의 복잡한 규칙 체계

## ✨ 주요 기능

- 📸 **가계부 이미지 OCR**: Gemini Vision API를 활용한 정확한 텍스트 추출
- 🔍 **지능형 품목 분석**: LLM을 통한 품목명 교정 및 의미 파악
- 🎯 **정확한 코드 매칭**: 벡터 유사도 + 키워드 검색 하이브리드 방식
- 📊 **신뢰도 기반 분류**: DEFINITE(확실) / AMBIGUOUS(모호) 구분
- 📋 **입력코드별 집계**: 분류된 항목의 수입/지출 자동 합산

## 🛠 기술 스택

### Core Technologies
- **Python** 3.9+
- **Streamlit** 1.28+ - 웹 인터페이스
- **LangChain** 0.1+ - RAG 파이프라인 구성
- **FAISS** - 벡터 유사도 검색
- **Pandas** - 데이터 처리

### AI/ML Models
- **OpenAI GPT-4o** - 품목 분석 및 분류 추론
- **OpenAI text-embedding-3-large** - 3072차원 벡터 임베딩
- **Google Gemini 1.5 Flash** - 이미지 OCR

### Data Processing
- **PyMuPDF** - PDF 파싱
- **pdfplumber** - 복잡한 PDF 표 추출
- **SemanticChunker** - 의미 단위 텍스트 분할

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/monodani/CateGOMe.git
cd CateGOMe

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`.streamlit/secrets.toml` 파일 생성:
```toml
OPENAI_API_KEY = "sk-..."
GENAI_API_KEY = "AI..."
```

### 3. 데이터 준비

벡터스토어가 이미 구축되어 있다면 건너뛸 수 있습니다.

```bash
# 벡터스토어 디렉토리 구조 확인
vectorstores/
├── cases/          # 조사사례집 벡터스토어
│   ├── index.faiss
│   └── index.pkl
└── classification/ # 항목분류집 벡터스토어
    ├── index.faiss
    └── index.pkl

# CSV 데이터 확인
data/classification_code.csv
```

### 4. 실행

```bash
streamlit run app.py
```

## 📊 데이터 구조

### 입력 데이터
- **가계부 이미지**: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF 지원
- 이미지 내 표 형식: 품목명, 수입, 지출 열 포함

### 분류 기준 데이터

| 데이터소스 | 설명 | 우선순위 |
|-----------|------|---------|
| 조사사례집 | 예외 사례 및 구체적 지침 | 1순위 |
| 항목분류집 | 일반적인 분류 기준 | 2순위 |
| Q&A | 통계청 내부 질의응답 | 1순위 |

### 출력 형식

#### DEFINITE (명확한 분류)
```json
{
  "classification_type": "DEFINITE",
  "result": {
    "input_code": "1085",
    "confidence": "95%",
    "reason": "케이크 전문점 구매 조각케이크",
    "evidence": "출처: 조사사례집..."
  }
}
```

#### AMBIGUOUS (모호한 분류)
```json
{
  "classification_type": "AMBIGUOUS",
  "candidates": [
    {"input_code": "1747", "confidence": "60%"},
    {"input_code": "1740", "confidence": "40%"}
  ]
}
```

## 🏗 아키텍처

상세 구조는 [architecture.md](architecture.md) 참조

### 핵심 처리 흐름
1. 📸 이미지 업로드 → Gemini OCR
2. 🔍 품목 추출 → LLM 품목명 정제
3. 📚 컨텍스트 검색 → 벡터 + 키워드 하이브리드
4. 🤖 분류 추론 → GPT-4o 체인
5. 📊 결과 집계 → 코드별 합산

## 🌐 배포 (Streamlit Cloud)

### 성능 고려사항
- **동시 사용자**: 5-10명 권장 (Streamlit Cloud 무료 티어 기준)
- **메모리 제한**: 1GB RAM (Community Cloud)
- **처리 시간**: 품목 10개 기준 약 30-60초

### 배포 설정
1. GitHub 저장소 연결
2. Streamlit Cloud 앱 생성
3. Secrets 설정 (OPENAI_API_KEY, GENAI_API_KEY)
4. Python 3.9 런타임 선택

## 📝 운영 및 유지보수 가이드

### 일일 점검사항
- API 키 유효성 및 사용량 모니터링
- 벡터스토어 무결성 확인
- 에러 로그 검토

### 월간 업데이트
- 신규 분류코드 CSV 업데이트
- 조사사례집/항목분류집 변경사항 반영
- 벡터스토어 재구축 (필요시)

### 벡터스토어 재구축
```bash
# 조사사례집 재구축
python scripts/build_cases_vectorstore.py

# 항목분류집 재구축  
python scripts/build_classification_vectorstore.py
```

## 🔧 트러블슈팅

| 문제 | 원인 | 해결방법 |
|------|------|----------|
| OCR 실패 | 이미지 품질 불량 | 고해상도 이미지 사용 |
| 분류 오류 | 벡터스토어 미갱신 | 최신 분류집으로 재구축 |
| 느린 처리 | API 한도 도달 | API 키 교체 또는 대기 |

## 📄 라이선스

본 프로젝트는 통계청 내부 사용 목적으로 개발되었습니다.

## 👥 기여자

- 개발팀: 통계청 AI 적용 데이터사이언스(전문가 양성) 과정 5조(CateGOMe 프로젝트팀)
- 자문: 문성현 교수

## 📞 문의

기술 지원 및 문의사항은 프로젝트 관리자에게 연락 바랍니다.
