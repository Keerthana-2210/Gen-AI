"""Ask one question and print a document-grounded Gemini answer."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import answer_question


def main() -> int:
    query = input("Enter your question:\n").strip()
    if not query:
        print("Please enter a non-empty question.")
        return 0

    try:
        result = answer_question(query)
    except FileNotFoundError as error:
        print(f"Vector database unavailable: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    if result["sources"]:
        for reference in result["sources"]:
            print(f"- {reference['source']}, Page {reference['page']}")
    else:
        print("- No source references found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
