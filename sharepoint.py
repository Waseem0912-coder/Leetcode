extract_prompt_guarded = PromptTemplate(
    input_variables=["topic", "context"],
    template_format="jinja2",
    template=(
        "You extract grounded bullet points.\n"
        "Base your answer ONLY on the provided context. Do NOT invent facts.\n"
        "If no relevant information is found, respond ONLY with [].\n\n"
        "TOPIC: {topic}\n\n"
        "CONTEXT:\n{context}\n\n"
        "Return ONLY a JSON list of objects. Each object MUST have:\n"
        "- 'bullet': a short standalone paraphrase grounded in the context\n"
        "- 'evidence': a list of 1–3 short quotes COPIED VERBATIM from the context that justify the bullet\n\n"
        "{% raw %}Example (JSON array):\n"
        "[{\"bullet\": \"Budget increased by 10% in Q4.\", \"evidence\": [\"budget increased by 10% in Q4\"]}]\n"
        "{% endraw %}"
    ),
)
