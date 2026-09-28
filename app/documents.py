"""Load the synthetic documents and personas. Each '## ' section of a document becomes one chunk.

Why chunks? Models work best with small, focused pieces of text, and search can point at the
exact section that answers a question. Here the documents already have clean sections, so one
section = one chunk (no need for fixed-size text windows).
"""

from pathlib import Path

import yaml

from app.models import Chunk, Persona

# Python note: `Path(__file__)` is this file's path; `.resolve().parent.parent` climbs from
# app/documents.py up to the project root, and `/ "data"` joins paths (like path.join in Node).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_KEYS = ("doc_id", "version", "title", "status", "effective_date", "classification",
                 "access", "source_type")


def load_chunks(docs_dir: Path = DATA_DIR / "docs") -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):  # every Markdown file, in a stable order
        # File layout: ---\n<YAML front matter>\n---\n# Title\n\n## Section\n...
        # Splitting on "---" at most twice gives 3 parts: "" (before the first ---), the YAML
        # metadata, and the body. Python lets you unpack them into 3 variables at once;
        # `_` is the convention for "I don't need this part".
        _, front_matter, body = path.read_text(encoding="utf-8").split("---", 2)
        meta = yaml.safe_load(front_matter)  # YAML text -> Python dict (like JSON.parse)

        # body.split("\n## ") cuts the body at every section heading. [1:] skips the first piece
        # (the "# Title" line before the first section). enumerate(..., start=1) yields
        # (1, section), (2, section), ... so every chunk gets a section number.
        for number, section in enumerate(body.split("\n## ")[1:], start=1):
            # partition splits at the first newline: the heading, the newline, and the rest.
            heading, _, text = section.partition("\n")
            chunks.append(
                Chunk(
                    # Python note: f"..." is a template literal, like `${a}@v${b}` in JS.
                    chunk_id=f"{meta['doc_id']}@v{meta['version']}#{number}",
                    section=heading.strip(),
                    text=text.strip(),
                    # Copy the metadata fields from the front matter into the chunk.
                    # Python note: `**{...}` spreads a dict into named arguments, like {...obj}.
                    **{key: meta[key] for key in METADATA_KEYS},
                )
            )
    return chunks


def load_personas(path: Path = DATA_DIR / "personas.yaml") -> dict[str, Persona]:
    # Python note: a dict comprehension, i.e. build {id: Persona} in one line. Equivalent to
    # Object.fromEntries(list.map(p => [p.id, new Persona(p)])) in JS.
    return {p["id"]: Persona(**p) for p in yaml.safe_load(path.read_text(encoding="utf-8"))}
