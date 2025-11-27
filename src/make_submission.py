"""
make_submission.py

agents.txt에 적힌 파일명을 기준으로 submission_*.py 파일을 생성한다.
폴더 구조가 변해도 파일명을 검색해 올바른 경로를 찾아 생성한다.
"""

from pathlib import Path

# Kaggle wrapper를 뒤에 덧붙여 제출용 에이전트를 만든다.
WRAPPER = """

# ================= Kaggle 제출용 엔트리 함수 =================
def agent(observation, configuration):
    \"\"\"Kaggle에서 요구하는 기본 에이전트 함수.\"\"\"
    return my_agent(observation, configuration)
"""

BASE_DIR = Path(__file__).resolve().parent  # src 디렉터리
TXT_FILE = BASE_DIR / "agents.txt"


def read_from_txt() -> list[str]:
    """agents.txt에서 줄 단위로 파일명을 읽어 리스트로 반환."""
    if not TXT_FILE.exists():
        return []

    lines = TXT_FILE.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def resolve_agent_path(agent_entry: str) -> Path | None:
    """
    agents.txt에 적힌 항목을 실제 파일 경로로 해석한다.
    - 절대경로면 그대로 사용
    - BASE_DIR 하위의 상대경로면 그대로 사용
    - 파일명만 적혔다면 BASE_DIR 이하를 검색해 첫 번째 일치 항목을 사용
    """
    entry_path = Path(agent_entry)

    # 확장자가 없으면 .py로 보정
    if entry_path.suffix == "":
        entry_path = entry_path.with_suffix(".py")

    if entry_path.is_absolute():
        return entry_path if entry_path.exists() else None

    direct = (BASE_DIR / entry_path).resolve()
    if direct.exists():
        return direct

    matches = list(BASE_DIR.rglob(entry_path.name))
    if not matches:
        return None

    if len(matches) > 1:
        first = matches[0]
        rel = first.relative_to(BASE_DIR)
        print(f"[WARN] '{agent_entry}' 후보가 여러 개입니다. 첫 경로 사용: {rel}")
        return first

    return matches[0]


def create_submission(src_path: Path) -> None:
    """소스 파일과 같은 폴더에 submission_<이름>.py를 생성."""
    if not src_path.exists():
        print(f"[ERROR] 파일을 찾을 수 없음: {src_path}")
        return

    dst_path = src_path.parent / f"submission_{src_path.stem}.py"
    code = src_path.read_text(encoding="utf-8")
    dst_path.write_text(code + WRAPPER, encoding="utf-8")

    rel = dst_path.relative_to(BASE_DIR)
    print(f"[OK] 생성 완료: {rel}")


def main() -> None:
    agents = read_from_txt()

    if not agents:
        print("[ERROR] agents.txt에 에이전트 목록이 없습니다. 파일을 확인하세요.")
        return

    print("처리할 에이전트 목록:", agents)

    for agent_entry in agents:
        src_path = resolve_agent_path(agent_entry)
        if src_path is None:
            print(f"[ERROR] 대상 파일을 찾지 못했습니다: {agent_entry}")
            continue
        create_submission(src_path)


if __name__ == "__main__":
    main()
