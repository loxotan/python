너는 제품 기획자, UX 디자이너, 풀스택 개발 리드, AI 앱 아키텍트 역할을 동시에 수행한다.

나는 “TasteMap / 취향지도”라는 앱을 만들고 있다.

핵심 철학:
“나는 누구인가?”라는 질문에 대해, 나를 나로 만드는 것은 내가 무엇을 좋아하고 싫어하는지이며, 그것을 구조화해서 보는 것이 자기이해의 출발점이다.

앱의 핵심 개념:
사용자는 자신의 호/불호를 기록한다.
사용자가 직접 입력하는 것은 최대한 적어야 한다.
기본 입력은 딱 두 가지다.

1. 항목 또는 문장
2. 1~5점 평점

예:
- “곱창전골” / 5점
- “오늘 분위기 좋은 노포 곱창전골집에서 애인이랑 소주 한잔 했음” / 5점
- “맥도날드에서 1955버거 먹었는데 나쁘지 않았음” / 3점
- “카뮈의 결혼, 여름을 읽고 많은 생각이 들었다” / 5점
- “교수님 앞에서 발표하는 건 너무 긴장돼서 싫다” / 2점

앱은 AI를 이용해서 자동으로 다음 정보를 생성한다.

- 주 카테고리
- 맥락 카테고리
- 서브카테고리
- 태그
- 선호 신호
- 신뢰도
- 나중에 추천에 쓸 수 있는 취향 데이터

예시 출력:

입력:
“오늘 분위기 좋은 노포 곱창전골집에서 애인이랑 소주 한잔 했음”
평점: 5

AI 분류 결과:
{
  "primary_category": "음식",
  "context_categories": ["장소", "관계", "상황"],
  "subcategories": ["한식", "전골", "내장요리", "술안주"],
  "tags": ["곱창전골", "곱창", "내장", "국물", "진한맛", "노포", "소주", "술자리", "애인", "정겨운 분위기"],
  "preference_signal": "진한 한식 전골과 노포 술자리 분위기에 대한 긍정적 선호로 보입니다.",
  "confidence": 0.9
}

앱의 장기 목표:
1. 호/불호 기록
2. AI 자동 카테고리/태그 생성
3. 사용자가 분류 결과를 저장 전 수정 가능
4. 카테고리별/키워드별 마인드맵
5. 나를 표현하는 형용사 5개 생성
6. 취향 기반 새로운 음식/장소/활동/콘텐츠 추천
7. 월별/연도별 취향 변화 타임라인
8. 주변인물의 호/불호 기록
9. 기본 비공개, 명시적 동의 시에만 공유
10. 광고 없음, 추후 월구독 가능성

현재 개발 상태:
Python + Streamlit + SQLite로 프로토타입을 만들고 있다.

현재 구현된 기능:
- raw_text 입력
- rating 1~5 입력
- OpenAI API 기반 AI 분류
- API 실패 시 rule-based fallback
- 주 카테고리 / 맥락 카테고리 / 서브카테고리 / 태그 / 선호 신호 / 신뢰도 표시
- SQLite 저장
- 기록 목록 표시
- 인기 태그 Top 10
- 카테고리별 평균 평점

현재 기술 스택:
- Python 3.11+
- Streamlit
- SQLite
- Pydantic
- OpenAI API
- python-dotenv
- Claude Code 사용 가능
- VS Code 사용
- Git 사용 가능

현재 프로젝트 예상 구조:
tastemap/
  app.py
  db.py
  models.py
  ai_classifier.py
  analysis.py
  recommender.py
  requirements.txt
  .gitignore
  .env
  data/tastemap.db

내가 원하는 결과물:
이 아이디어를 제품 기획서와 개발 계획으로 정리해줘.

아래 항목을 반드시 포함해라.

1. 제품 정의
- 이 앱이 무엇인지 한 문장으로 정의
- 핵심 사용자 가치
- 기존 일기앱, 무드트래커, 노트앱과의 차별점

2. 핵심 사용자 시나리오
- 첫 사용자 온보딩
- 하루 기록 플로우
- AI 분류 결과 확인/수정 플로우
- 나의 취향 요약 보기
- 추천 받기
- 월간 리포트 보기

3. MVP 범위
현재 프로토타입을 기준으로 v0.1, v0.2, v0.3, v1.0을 나눠줘.

반드시 다음 우선순위를 반영해라.
- v0.1: 입력, AI 분류, 저장, 목록
- v0.2: 저장 전 분류 결과 수정
- v0.3: 태그 타입 분리, 좋아요/싫어요 분석
- v0.4: 추천 기능
- v0.5: 마인드맵
- v1.0: Android 또는 iOS 앱화

4. 데이터 모델 설계
아래 개념을 포함해서 설계해라.
- PreferenceEntry
- PreferenceClassification
- Tag
- TagType
- PersonProfile
- MonthlySummary
- Recommendation

각 모델에 필요한 필드를 제안해라.

5. AI 분류 프롬프트 설계
OpenAI 또는 Claude API에 넣을 시스템 프롬프트와 JSON schema를 작성해라.
조건:
- 한국어 입력 지원
- JSON only
- 성격 단정 금지
- 정신건강 진단 금지
- 민감한 속성 추론 금지
- 관찰 가능한 선호 신호만 분류
- 모호하면 미분류와 낮은 confidence 사용
- rating 4~5는 긍정 선호
- rating 1~2는 부정 선호
- rating 3은 중립 또는 불명확

6. Claude Code 개발 태스크 목록
Claude Code에 바로 던질 수 있는 작은 단위의 태스크로 쪼개라.
각 태스크는 다음 형식으로 작성해라.

Task 이름:
목표:
수정 파일:
구현 내용:
테스트 방법:
Claude Code 프롬프트:

특히 아래 태스크를 포함해라.
- 저장 전 AI 분류 결과 수정 UI
- 태그 타입 분리
- 좋아요 태그 / 싫어요 태그 분석
- 월별 취향 요약
- 추천 엔진 v0
- pyvis 또는 networkx 기반 마인드맵
- CSV/JSON 백업
- API 실패 디버그 표시
- Streamlit UI 정리

7. MCP 연동 계획
Claude Code와 MCP를 연동한다고 가정하고, 어떤 MCP 서버를 쓰면 좋은지 제안해라.

후보:
- filesystem MCP: 프로젝트 파일 읽기/쓰기
- git MCP: 변경사항 확인, 커밋 로그 관리
- sqlite MCP: tastemap.db 직접 질의
- browser/playwright MCP: Streamlit 앱 UI 테스트
- memory 또는 notes MCP: 제품 요구사항, 결정사항 저장
- fetch/web MCP: 문서 참고

각 MCP가 이 프로젝트에서 어떤 역할을 하는지 설명해라.

8. MCP 기반 개발 워크플로
다음 루틴을 만들어라.
- 요구사항 작성
- Claude Code가 plan 작성
- 코드 수정
- 앱 실행
- UI 테스트
- SQLite 확인
- Git diff 확인
- 커밋
- 다음 태스크 진행

9. Claude Code용 최상위 지침 파일
프로젝트 루트에 둘 CLAUDE.md 내용을 작성해라.
내용:
- 프로젝트 목적
- 금지사항
- 코드 스타일
- 파일 역할
- AI 분류 안전 규칙
- 테스트 방법
- 커밋 전 확인사항

10. 제품 리스크
아래 리스크를 분석해라.
- 기록 피로
- AI 분류 오류
- 태그 난잡화
- 개인정보
- 주변인물 기록의 윤리 문제
- API 비용
- 앱스토어/플레이스토어 확장성
- 사용자가 오래 쓸 이유 부족

각 리스크에 대한 대응책을 제안해라.

11. 최종 산출물
다음 형식으로 정리해라.
- Product Brief
- MVP Roadmap
- Data Model
- AI Prompt
- Claude Code Task Backlog
- MCP Setup Plan
- CLAUDE.md Draft
- Next 7 Days Action Plan

중요한 제약:
- 현재 단계에서는 Python + Streamlit 프로토타입 중심으로 작성한다.
- 모바일 앱 코드는 아직 작성하지 않는다.
- 로그인, 결제, 클라우드 동기화는 v1.0 이후로 미룬다.
- 너무 거창하게 만들지 말고, 실제로 혼자 개발 가능한 범위로 줄여라.
- 모든 개발 태스크는 작고 검증 가능해야 한다.