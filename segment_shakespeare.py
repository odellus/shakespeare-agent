import os
import re
import json
from pathlib import Path

INPUT_FILE = "shakespeare.txt"
OUTPUT_DIR = "plays"


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def extract_toc(text: str) -> list[str]:
    """Pull the ordered list of work titles from the table of contents."""
    entries = []
    in_toc = False
    for line in text.split("\n"):
        if line.strip() == "Contents":
            in_toc = True
            continue
        if in_toc:
            if line.startswith("    ") and len(line.strip()) > 3:
                entries.append(line.strip())
            elif line.strip() and not line.startswith("    "):
                break
    return entries


def find_play_boundaries(text: str, titles: list[str]) -> list[tuple[str, int, int]]:
    """
    For every TOC title find the start position of the actual play text.
    Returns [(title, start, end), ...] in file order.
    """
    # We only look after the TOC ends.  The TOC ends around line 100,
    # but the first actual play (THE SONNETS) starts around line 102.
    # We'll search from a safe offset.
    search_start = text.find("\nTHE SONNETS\n")
    if search_start == -1:
        search_start = 0

    boundaries = []
    for title in titles:
        # Try exact match first, then fuzzy
        pattern = re.escape(title)
        m = re.search(rf"^{pattern}$", text[search_start:], re.MULTILINE)
        if not m:
            # Smart-quote → ASCII apostrophe fallback
            alt = title.replace("\u2019", "'")
            pattern = re.escape(alt)
            m = re.search(rf"^{pattern}$", text[search_start:], re.MULTILINE)

        if m:
            abs_pos = search_start + m.start()
            boundaries.append((title, abs_pos))
        else:
            print(f"  WARNING: could not find start of '{title}'")

    # Sort by position and compute end = next start
    boundaries.sort(key=lambda x: x[1])
    result = []
    for i, (title, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
        result.append((title, start, end))

    return result


def clean_speech(text: str) -> str:
    """Remove stage directions and extraneous whitespace."""
    # Remove [_Stage direction._] patterns
    text = re.sub(r"\[_[^\]]+_\]", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_speakers(play_text: str) -> dict[str, list[str]]:
    """
    Extract every speaker block from a play.
    Returns {speaker_name: [speech1, speech2, ...], ...}
    """
    blocks = re.findall(
        r"^([A-Z][A-Z\s',]+)\.\n(.*?)(?=\n[A-Z][A-Z\s',]+\.\n|\Z)",
        play_text,
        re.DOTALL | re.MULTILINE,
    )

    speeches = {}
    for speaker, speech in blocks:
        name = speaker.strip()
        # Skip stage-direction-like "speakers" that aren't real characters
        if name in {"SCENE", "ACT", "ENTER", "EXIT", "EXUENT", "THE END"}:
            continue
        cleaned = clean_speech(speech)
        if cleaned:
            speeches.setdefault(name, []).append(cleaned)

    return speeches


def main():
    print("Loading text...")
    text = load_text(INPUT_FILE)

    print("Extracting TOC...")
    titles = extract_toc(text)
    print(f"  → {len(titles)} works listed")

    print("Finding play boundaries...")
    boundaries = find_play_boundaries(text, titles)
    print(f"  → located {len(boundaries)} works in text")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = {}

    for title, start, end in boundaries:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        play_text = text[start:end].strip()

        print(f"\n{title} ({len(play_text):,} chars)")
        speakers = parse_speakers(play_text)
        print(f"  → {len(speakers)} speakers")

        play_dir = os.path.join(OUTPUT_DIR, slug)
        os.makedirs(play_dir, exist_ok=True)

        # Save raw play text
        with open(os.path.join(play_dir, "_play.txt"), "w", encoding="utf-8") as f:
            f.write(play_text)

        # Save each speaker's speeches (clean — no redundant prefix, no stage directions)
        speaker_names = []
        for speaker, speeches in sorted(speakers.items()):
            speaker_names.append(speaker)
            speaker_text = "\n\n---\n\n".join(speeches)
            with open(
                os.path.join(play_dir, f"{speaker}.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(speaker_text)

        manifest[slug] = {
            "title": title,
            "speakers": speaker_names,
            "chars": len(play_text),
        }

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Saved to ./{OUTPUT_DIR}/")
    print(f"Manifest: {len(manifest)} plays")


if __name__ == "__main__":
    main()
