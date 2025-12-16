import json
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent

app = FastAPI(
    title="Document Translation API",
    description="Translate JSON documents between languages using AI",
    version="0.1.0",
)


class TranslationRequest(BaseModel):
    """Request model for document translation."""

    document: dict[str, Any] = Field(
        ...,
        description="The JSON document to translate",
        examples=[{"title": "Hello World", "description": "A greeting message"}],
    )
    source_language: str = Field(
        default="auto",
        description="Source language (use 'auto' for automatic detection)",
        examples=["English", "Spanish", "auto"],
    )
    target_language: str = Field(
        ...,
        description="Target language for translation",
        examples=["French", "German", "Japanese"],
    )
    preserve_keys: bool = Field(
        default=True,
        description="Whether to preserve JSON keys (only translate values)",
    )


class TranslationResponse(BaseModel):
    """Response model for document translation."""

    translated_document: dict[str, Any] = Field(
        ...,
        description="The translated JSON document",
    )
    source_language: str = Field(
        ...,
        description="Detected or specified source language",
    )
    target_language: str = Field(
        ...,
        description="Target language used for translation",
    )


# Create the translation agent
translation_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="""You are a professional translator. Your task is to translate JSON documents.

IMPORTANT RULES:
1. Return ONLY valid JSON - no markdown, no explanations, no code blocks
2. Preserve the exact JSON structure (all keys and nesting)
3. Only translate string values, keep numbers, booleans, and null unchanged
4. Maintain the same JSON key names (do not translate keys unless specifically asked)
5. Preserve any placeholders like {name}, {{variable}}, %s, etc.
6. Keep technical terms, brand names, and proper nouns as appropriate
7. Ensure the translation sounds natural in the target language""",
)


@app.post("/translate", response_model=TranslationResponse)
async def translate_document(request: TranslationRequest) -> TranslationResponse:
    """
    Translate a JSON document from one language to another.

    The endpoint preserves the JSON structure and translates string values
    while keeping keys, numbers, booleans, and null values unchanged.
    """
    # Build the translation prompt
    key_instruction = (
        "DO NOT translate the JSON keys, only translate the string values."
        if request.preserve_keys
        else "Translate both keys and values."
    )

    source_lang = (
        "Automatically detect the source language"
        if request.source_language == "auto"
        else f"Source language: {request.source_language}"
    )

    prompt = f"""Translate this JSON document to {request.target_language}.

{source_lang}
{key_instruction}

JSON document to translate:
{json.dumps(request.document, ensure_ascii=False, indent=2)}

Return ONLY the translated JSON object, nothing else."""

    try:
        result = await translation_agent.run(prompt)
        translated_text = result.data.strip()

        # Clean up potential markdown code blocks
        if translated_text.startswith("```"):
            lines = translated_text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's closing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            translated_text = "\n".join(lines)

        translated_doc = json.loads(translated_text)

        # Detect source language from response if auto
        detected_source = (
            request.source_language
            if request.source_language != "auto"
            else "auto-detected"
        )

        return TranslationResponse(
            translated_document=translated_doc,
            source_language=detected_source,
            target_language=request.target_language,
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse translated document as JSON: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}",
        )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
