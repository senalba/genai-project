#!/usr/bin/env python3
# summarize_file.py
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIError, AuthenticationError


def call_summary(
    client: OpenAI,
    model: str,
    text: str,
    max_words: int,
    retries: int = 3,
    backoff: float = 1.5,
) -> str:
    prompt = (
        f"Summarize the following text in up to {max_words} words. "
        f"Be concise and capture the key points:\n\n{text}"
    )
    for attempt in range(retries):
        try:
            resp = client.responses.create(model=model, input=prompt, temperature=0)
            # Prefer SDK helper
            summary = getattr(resp, "output_text", None)
            if summary:
                return summary.strip()
            # Fallback extraction
            try:
                return resp.output[0].content[0].text.strip()
            except Exception:
                return str(resp)
        except (RateLimitError, APIConnectionError, APIError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff**attempt)
        except AuthenticationError:
            raise
    raise RuntimeError("Unexpected failure with no exception raised.")


def write_batch_jsonl(input_paths, batch_jsonl_path, model, max_words):
    with open(batch_jsonl_path, "w", encoding="utf-8") as out:
        for p in input_paths:
            text = Path(p).read_text(encoding="utf-8")
            body = {
                "model": model,
                "input": (
                    f"Summarize the following text in up to {max_words} words. "
                    f"Be concise and capture the key points:\n\n{text}"
                ),
                "temperature": 0,
            }
            line = {
                "custom_id": Path(p).stem,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            out.write(json.dumps(line, ensure_ascii=False) + "\n")


def main():
    load_dotenv()

    p = argparse.ArgumentParser(description="Summarize text with OpenAI")
    p.add_argument(
        "input", help="Path to input text file (or directory with --batch-jsonl)"
    )
    p.add_argument("-o", "--output", help="Path to save the single-file summary")
    p.add_argument(
        "-m", "--model", default="gpt-4o-mini", help="Model (default: gpt-4o-mini)"
    )
    p.add_argument(
        "--max-words", type=int, default=200, help="Max words (default: 200)"
    )
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries on transient errors (default: 3)",
    )
    p.add_argument(
        "--batch-jsonl", help="Write Batch API JSONL here (input must be a directory)"
    )
    args = p.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env or environment")

    # Batch mode: create JSONL for /v1/batches
    if args.batch_jsonl:
        in_path = Path(args.input)
        if not in_path.is_dir():
            raise ValueError(
                "When using --batch-jsonl, INPUT must be a directory of text files."
            )
        files = sorted([str(p) for p in in_path.glob("**/*") if p.is_file()])
        if not files:
            raise ValueError("No files found in the input directory.")
        write_batch_jsonl(files, args.batch_jsonl, args.model, args.max_words)
        print(f"Wrote batch file: {args.batch_jsonl} ({len(files)} requests)")
        return

    # Single-file mode
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    client = OpenAI(api_key=api_key)
    try:
        summary = call_summary(
            client, args.model, text, args.max_words, retries=args.retries
        )
    except RateLimitError as e:
        sys.exit(f"Rate limit / quota hit: {e}")
    except AuthenticationError:
        sys.exit("Auth failed. Check OPENAI_API_KEY.")
    except (APIConnectionError, APIError) as e:
        sys.exit(f"API error: {e}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(summary + "\n")
    else:
        print(summary)


if __name__ == "__main__":
    main()
