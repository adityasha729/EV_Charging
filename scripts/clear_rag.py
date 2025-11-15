"""Script to clear the RAG index directory safely.

Usage:
    python scripts/clear_rag.py         # prompts for confirmation
    python scripts/clear_rag.py --yes   # runs without prompt

This removes files in the `rag_index` directory and recreates an empty meta.json placeholder.
"""
import os
import shutil
import json
import argparse

RAG_DIR = os.path.join(os.path.dirname(__file__), '..', 'rag_index')
RAG_DIR = os.path.abspath(RAG_DIR)
META_PATH = os.path.join(RAG_DIR, 'meta.json')

EMPTY_META = {'metadatas': [], 'model_name': ''}


def clear_rag(confirm: bool = True):
    if not os.path.exists(RAG_DIR):
        print(f"RAG directory does not exist at {RAG_DIR}. Nothing to clear.")
        return

    if confirm:
        answer = input(f"This will delete all files in {RAG_DIR}. Continue? [y/N]: ")
        if answer.lower() not in ('y', 'yes'):
            print('Aborted.')
            return

    # remove everything inside the rag dir
    for name in os.listdir(RAG_DIR):
        path = os.path.join(RAG_DIR, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            print(f"Failed to remove {path}: {e}")

    # ensure dir exists and write empty meta.json
    os.makedirs(RAG_DIR, exist_ok=True)
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(EMPTY_META, f, ensure_ascii=False, indent=2)

    print(f"Cleared RAG directory at {RAG_DIR} and wrote empty meta.json.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clear RAG index directory')
    parser.add_argument('--yes', action='store_true', help='Confirm without prompting')
    args = parser.parse_args()
    clear_rag(confirm=not args.yes)
