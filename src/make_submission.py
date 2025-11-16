"""
make_submission.py

에이전트 목록을 agents.txt에서 읽어서
여러 개의 submission_*.py 파일을 자동 생성하는 스크립트.
"""

import os
from pathlib import Path

# Kaggle wrapper
WRAPPER = """

# ================= Kaggle용 진입점 함수 =================
def agent(observation, configuration):
    \"\"\"Kaggle이 호출하는 기본 에이전트 함수.\"\"\"
    return my_agent(observation, configuration)
"""

BASE_DIR = Path(__file__).resolve().parent
TXT_FILE = BASE_DIR / "agents.txt"


def read_from_txt():
    """agents.txt에서 라인별로 파일名前 읽기"""
    if not TXT_FILE.exists():
        return []

    lines = TXT_FILE.read_text(encoding="utf-8").splitlines()
    agents = [line.strip() for line in lines if line.strip()]
    return agents


def create_submission(src_path: Path):
    """src_path 기반으로 submission_*.py 생성"""
    if not src_path.exists():
        print(f"[ERROR] 파일 없음: {src_path}")
        return

    dst_path = src_path.parent / f"submission_files/submission_{src_path.stem}.py"

    code = src_path.read_text(encoding="utf-8")
    dst_path.write_text(code + WRAPPER, encoding="utf-8")

    print(f"[OK] 생성됨 → {dst_path.name}")


def main():
    # txt에서 읽기
    agents = read_from_txt()

    if not agents:
        print("⚠️ 에이전트 목록이 없습니다.")
        print("agents.txt 파일을 확인하세요.")
        return

    print("처리할 에이전트 목록:", agents)

    for agent_file in agents:
        create_submission(BASE_DIR / agent_file)


if __name__ == "__main__":
    main()
