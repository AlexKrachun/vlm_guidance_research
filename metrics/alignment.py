import argparse
import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict


DEFAULT_MODEL = "gpt-4o-2024-05-13"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_eval_prompt(prompt: str) -> str:
    return f"""You are an assistant evaluating an image on two independent aspects:
(1) how well it aligns with the meaning of a given text prompt, and
(2) its visual quality.

The text prompt is: "{prompt}"

---

PART 1: PROMPT ALIGNMENT (Semantic Fidelity)
Evaluate only the meaning conveyed by the image - ignore visual artifacts.
Focus on:
- Are the correct objects present and depicted in a way that clearly demonstrates their intended roles and actions from the prompt?
- Does the scene illustrate the intended situation or use-case in a concrete and functional way, rather than through symbolic, metaphorical, or hybrid representation?
- If the described usage or interaction is missing or unclear, alignment should be penalized.
- Focus strictly on the presence, roles, and relationships of the described elements - not on rendering quality.

Score from 1 to 5:
5: Fully conveys the prompt's meaning with correct elements
4: Mostly accurate - main elements are correct, with minor conceptual or contextual issues
3: Main subjects are present but important attributes or actions are missing or wrong
2: Some relevant components are present, but key elements or intent are significantly misrepresented
1: Does not reflect the prompt at all

---

PART 2: VISUAL QUALITY (Rendering Fidelity)
Now focus only on how the image looks visually - ignore whether it matches the prompt.
Focus on:
- Are there rendering artifacts, distortions, or broken elements?
- Are complex areas like faces, hands, and shaped objects well-formed and visually coherent?
- Are complex areas like faces, hands, limbs, and object grips well-formed and anatomically correct?
- Is lighting, texture, and perspective consistent across the scene?
- Do elements appear physically coherent - i.e., do objects connect naturally (no floating tools, clipped limbs, or merged shapes)?
- Distortion, warping, or implausible blending of objects (e.g. melted features, fused geometry) should reduce the score.
- Unusual or surreal objects are acceptable if they are clearly rendered and visually deliberate.

Score from 1 to 5:
5: Clean, realistic, and fully coherent - no visible flaws
4: Mostly clean with minor visual issues or stiffness
3: Noticeable visual flaws (e.g. broken grips, distorted anatomy), but the image is still readable
2: Major visual issues - warped or broken key elements disrupt coherence
1: Severe rendering failure - image appears nonsensical or corrupted

---

Respond using this format:
### ALIGNMENT SCORE: score
### ALIGNMENT EXPLANATION: explanation
### QUALITY SCORE: score
### QUALITY EXPLANATION: explanation"""


def parse_evaluation_text(text: str) -> Dict[str, object]:
    patterns = {
        "alignment score": r"###\s*ALIGNMENT SCORE:\s*(\d+)",
        "alignment explanation": r"###\s*ALIGNMENT EXPLANATION:\s*(.*?)\s*###\s*QUALITY SCORE:",
        "quality score": r"###\s*QUALITY SCORE:\s*(\d+)",
        "quality explanation": r"###\s*QUALITY EXPLANATION:\s*(.*)",
    }

    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Failed to parse `{key}` from model response:\n{text}")
        value = match.group(1).strip()
        parsed[key] = int(value) if "score" in key else value
    return parsed


def evaluate_image_with_gpt(
    image_path: str,
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> Dict[str, object]:
    base64_image = encode_image(image_path)
    eval_prompt = build_eval_prompt(prompt)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": eval_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    request = urllib.request.Request(
        OPENAI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: {exc.code} {error_body}") from exc

    try:
        text = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenAI API response:\n{json.dumps(response_data, indent=2)}") from exc

    output = parse_evaluation_text(text)
    output["raw response"] = text
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute image-prompt alignment score using the SAP repository evaluation protocol."
    )
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--prompt", required=True, help="Prompt to evaluate against the image.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Vision-capable OpenAI model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens for the evaluator response.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full parsed result as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise SystemExit("OpenAI API key is required. Pass --api-key or set OPENAI_API_KEY.")

    result = evaluate_image_with_gpt(
        image_path=args.image,
        prompt=args.prompt,
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"Alignment score: {result['alignment score']}")
    print(f"Alignment explanation: {result['alignment explanation']}")
    print(f"Quality score: {result['quality score']}")
    print(f"Quality explanation: {result['quality explanation']}")


if __name__ == "__main__":
    main()


"""
python3 alignment_score.py \
  --image images/sap_bear.jpg \
  --prompt "A bear is performing a handstand in the park" \
  --api-key "$OPENAI_API_KEY"

"""