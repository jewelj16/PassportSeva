import json
import re
from pathlib import Path
from pypdf import PdfReader


def log(msg):
    print(f"[INFO] {msg}")


class DocumentLoader:
    def __init__(self, *raw_dirs: str):
        if not raw_dirs:
            raw_dirs = ("data/raw",)
        self.raw_dirs = [Path(d) for d in raw_dirs]

    def load_all(self) -> list:
        docs = []
        for raw_dir in self.raw_dirs:
            if not raw_dir.exists():
                print(f"[ERROR] Raw data directory not found: {raw_dir}")
                continue
            for file in sorted(raw_dir.iterdir()):
                if file.suffix == ".txt":
                    docs.extend(self._load_txt(file))
                elif file.suffix == ".pdf":
                    docs.extend(self._load_pdf(file))
                elif file.suffix == ".jsonl":
                    docs.extend(self._load_jsonl(file))
            log(f"Loaded docs from {raw_dir}")
        log(f"Total: {len(docs)} raw documents from {len(self.raw_dirs)} directories")
        return docs

    def _load_txt(self, path: Path) -> list:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = self._clean(text)
        if len(text) < 50:
            return []
        return [{"text": text, "source": path.name, "type": "article", "category": "general"}]

    def _load_pdf(self, path: Path) -> list:
        try:
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = self._clean(page.extract_text() or "")
                if len(text) > 50:
                    pages.append({
                        "text": text,
                        "source": f"{path.name}:page{i+1}",
                        "type": "pdf",
                        "category": "general"
                    })
            return pages
        except Exception as e:
            print(f"[WARNING] Failed to read PDF {path.name}: {e}")
            return []

    def _load_jsonl(self, path: Path) -> list:
        items = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "question" in obj and "answer" in obj:
                        text = f"Q: {obj['question']}\nA: {obj['answer']}"
                    elif "text" in obj:
                        text = obj["text"]
                    else:
                        continue
                    items.append({
                        "text": text,
                        "source": obj.get("source", path.name),
                        "type": "qa",
                        "category": obj.get("category", "general")
                    })
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Skipping malformed JSON at {path.name}:{line_num} — {e}")
        return items

    def _clean(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()
