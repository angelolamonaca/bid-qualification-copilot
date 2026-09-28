"""Load the synthetic documents and personas. Each '## ' section of a document becomes one chunk."""

from pathlib import Path

import yaml

from app.models import Chunk, Persona

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_KEYS = ("doc_id", "version", "title", "status", "effective_date", "classification",
                 "access", "source_type")


def load_chunks(docs_dir: Path = DATA_DIR / "docs") -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        # File layout: ---\n<YAML front matter>\n---\n# Title\n\n## Section\n...
        _, front_matter, body = path.read_text(encoding="utf-8").split("---", 2)
        meta = yaml.safe_load(front_matter)
        for number, section in enumerate(body.split("\n## ")[1:], start=1):
            heading, _, text = section.partition("\n")
            chunks.append(
                Chunk(
                    chunk_id=f"{meta['doc_id']}@v{meta['version']}#{number}",
                    section=heading.strip(),
                    text=text.strip(),
                    **{key: meta[key] for key in METADATA_KEYS},
                )
            )
    return chunks


def load_personas(path: Path = DATA_DIR / "personas.yaml") -> dict[str, Persona]:
    return {p["id"]: Persona(**p) for p in yaml.safe_load(path.read_text(encoding="utf-8"))}
