"""Ask the copilot a question from the terminal, as one of the simulated personas.

Run from the project root:
    uv run python -m scripts.ask mia "Do we meet the mandatory requirements of tender RVA-2026-014?"
"""

import sys

from app.config import get_settings
from app.documents import load_chunks, load_personas
from app.llm import AzureOpenAILLM
from app.qa import answer_question
from app.retrieval import SearchIndex

persona_id, question = sys.argv[1], sys.argv[2]
llm = AzureOpenAILLM(get_settings())
index = SearchIndex(load_chunks(), llm)
result = answer_question(question, load_personas()[persona_id], index, llm)

print(f"\nANSWER\n{result.answer}\n\nCITATIONS")
for source in result.citations:
    print(f"  - {source.title} v{source.version} > {source.section}  [{source.chunk_id}]")
print("\nLIMITATIONS")
for limitation in result.limitations or ["(none)"]:
    print(f"  - {limitation}")
trace = result.retrieval
print(f"\nSEARCH  {trace.searched_chunks} chunks searched; excluded {trace.excluded_no_permission} "
      f"(no permission) and {trace.excluded_superseded} (superseded)")
print(f"COST    {sum(u.input_tokens + u.output_tokens for u in result.usage)} tokens, "
      f"${sum(u.cost_usd for u in result.usage):.4f}, "
      f"{sum(u.latency_ms for u in result.usage) / 1000:.1f} s")
