import fitz  # PyMuPDF
import re
import json
import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel


# Used in detect_headings as a visual-ambiguity fallback:
# if a span is bold or large AND starts with one of these, it's likely a heading.
# Covers every major section type across ML/NLP/CV/Systems/Math papers.
SECTION_KEYWORDS = [
    # Core structure
    "abstract", "synopsis",
    "introduction", "contributions", "motivation", "overview",
    "problem statement", "problem definition", "research question",
    "paper organization", "paper outline",
    # Related work / survey
    "related work", "prior work", "prior art", "literature review",
    "literature survey", "previous work", "existing methods",
    "state of the art", "state-of-the-art",
    # Background / setup
    "background", "preliminaries", "preliminary",
    "notation", "notations", "definitions",
    "problem setup", "formal setup", "setup",
    "mathematical background", "theoretical background",
    # Methodology (broad)
    "methodology", "method", "methods",
    "approach", "proposed", "our method", "our model",
    "model", "architecture", "framework", "algorithm", "algorithms",
    "system", "technique", "formulation",
    "theorem", "proof", "lemma", "corollary", "proposition",
    "theory", "analysis", "complexity",
    "training", "optimization", "objective", "loss",
    "network", "encoder", "decoder", "attention", "transformer",
    "diffusion", "generative", "autoencoder",
    "embedding", "tokenization", "prompting",
    "fairness", "causal", "counterfactual", "inference",
    "probabilistic", "bayesian",
    # Datasets
    "dataset", "datasets", "data collection", "corpus",
    "benchmark", "annotation", "preprocessing", "augmentation",
    # Experiments
    "experiment", "experiments", "experimental",
    "evaluation", "implementation", "baseline", "baselines",
    "ablation", "simulation", "illustration", "case study",
    "hyperparameter", "comparison",
    # Results
    "result", "results", "performance", "finding", "findings",
    "accuracy", "outcome",
    # Discussion
    "discussion", "limitation", "limitations",
    "broader impact", "ethical", "ethics", "societal",
    "failure", "interpretation", "interpretability",
    # Conclusion
    "conclusion", "conclusions", "concluding",
    "future work", "future directions", "summary", "outlook",
    # Supplemental
    "appendix", "supplement", "acknowledgment", "acknowledgements",
    "reference", "references", "bibliography",
    # Math / CS theory
    "construction", "existence", "uniqueness", "decomposition",
    "representation", "structure", "characterization",
    "convergence", "approximation", "bound",
    "operator", "topology", "measure",
]


def extract_text_blocks(pdf_path: str) -> list[dict]:
    """Extract text blocks with font size info from PDF."""
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:  # skip images
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    blocks.append({
                        "text": span["text"].strip(),
                        "size": round(span["size"], 1),
                        "page": page_num + 1,
                        "bold": "Bold" in span.get("font", "")
                    })
    doc.close()
    return blocks


def detect_headings(blocks: list[dict]) -> list[dict]:
    """
    Detect headings based on font size, bold, and keyword signals.

    Key guards:
    - Minimum 4 chars: prevents DAG node labels (U, A, X, Y) from being headings
    - Maximum 120 chars: real headings are short; long lines are body text
    - Must be larger than body OR bold, AND match a keyword OR be short enough
    - Footnote/footer lines explicitly rejected (∗, †, conference stamps, URLs)
    """
    if not blocks:
        return []

    sizes = [b["size"] for b in blocks if b["text"]]
    body_size = max(set(sizes), key=sizes.count)  # most common font size = body

    # Patterns that indicate footer/footnote lines, not headings
    footer_patterns = [
        re.compile(r'^[∗†‡§¶]'),                          # footnote markers
        re.compile(r'conference|workshop|proceedings|arxiv', re.IGNORECASE),
        re.compile(r'https?://'),                           # URLs
        re.compile(r'equal contribution', re.IGNORECASE),
        re.compile(r'^\d{4}$'),                             # lone year
        re.compile(r'@'),                                   # email addresses
        # Figure/Table captions — not headings; block so their content stays in section
        re.compile(r'^(figure|table|fig\.)\s*\d', re.IGNORECASE),
    ]

    # Words that are NEVER real section headings even if bold/large.
    # These are word fragments or common terms that appear bold in-text
    # (e.g. "counter|factual" split by PyMuPDF across a line break produces
    # a standalone bold "factual" span that looks like a heading).
    # Also covers: bold terms mid-sentence, hyphenated word halves, etc.
    heading_blocklist = {
        # PDF line-break split artifacts from compound words
        "factual", "counterfactual", "ational", "tional", "ization",
        # Common bold in-text terms that are not section headings
        "note", "remark", "example", "claim", "step",
        "input", "output", "given", "such that", "where",
        "thus", "hence", "therefore", "however", "moreover",
        "furthermore", "additionally", "finally", "specifically",
        "formally", "intuitively", "importantly",
        # Numbers / letters that can appear bold as list labels
        "i", "ii", "iii", "iv", "v", "vi",
    }

    annotated = []
    for b in blocks:
        text = b["text"].strip()
        text_lower = text.lower()

        # Hard disqualifiers
        if len(text) < 4:                             # too short — DAG labels, symbols
            annotated.append({**b, "is_heading": False})
            continue
        if len(text) > 120:                           # too long — body text
            annotated.append({**b, "is_heading": False})
            continue
        if any(p.search(text) for p in footer_patterns):
            annotated.append({**b, "is_heading": False})
            continue
        if text_lower in heading_blocklist:            # never a real section heading
            annotated.append({**b, "is_heading": False})
            continue

        is_large = b["size"] > body_size + 1
        is_heading_keyword = any(text_lower.startswith(kw) for kw in SECTION_KEYWORDS)
        # A heading must be visually distinct (large or bold) AND either keyword-matched or short
        is_heading = (is_large or b["bold"]) and (is_heading_keyword or len(text) < 60)
        annotated.append({**b, "is_heading": is_heading})

    return annotated


def load_nougat():
    """Load Nougat processor and model once, reuse across calls."""
    print("    [Nougat] Loading model (first run downloads ~3GB)...")
    processor = NougatProcessor.from_pretrained("facebook/nougat-base")
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"    [Nougat] Model loaded on {device}")
    return processor, model, device


def extract_equations(pdf_path: str, processor=None, model=None, device="cpu") -> list[str]:
    """
    Extract equations from PDF using Nougat (facebook/nougat-base).

    Nougat is a vision transformer trained specifically on academic PDFs.
    It renders each page as an image and outputs academic markdown (LaTeX).
    We then parse that output to pull out LaTeX equation blocks.

    Why Nougat over regex:
    - PDFs store equations as glyphs/images — raw text extraction mangles them
    - Nougat sees the rendered page visually, same as a human would
    - Output is proper LaTeX (e.g. \\frac{}{}, \\sum_, \\mathbf{})
    """
    if processor is None or model is None:
        processor, model, device = load_nougat()

    doc = fitz.open(pdf_path)
    all_equations = []

    # Regex to extract equations from Nougat markdown output
    eq_patterns = [
        re.compile(r'\\\[(.*?)\\\]', re.DOTALL),                                        # \[ ... \]
        re.compile(r'\$\$(.*?)\$\$', re.DOTALL),                                        # $$ ... $$
        re.compile(r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}', re.DOTALL),      # equation env
        re.compile(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', re.DOTALL),            # align env
        re.compile(r'\\begin\{eqnarray\*?\}(.*?)\\end\{eqnarray\*?\}', re.DOTALL),     # eqnarray env
        # Inline math — 5 char min, but NOT re.DOTALL to prevent cross-line matching
        re.compile(r'\$([^$\n]{5,150}?)\$'),
        # Probability expressions — no DOTALL, capped at 200 chars to prevent paragraph capture
        re.compile(r'(P\s*\([^)]{1,80}\)\s*=\s*P\s*\([^)]{1,80}\))'),
        re.compile(r'(\\mathbb\{[A-Z]\}[^\n]{0,100}=[^\n]{0,100})(?=\n|$)'),
    ]

    for page_num, page in enumerate(doc):
        print(f"    [Nougat] Processing page {page_num + 1}/{doc.page_count}...")

        # Render page to image at 150 DPI (balance between quality and speed)
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Run Nougat inference
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=2048,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
            )
        page_markdown = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Parse equations from Nougat markdown
        for pattern in eq_patterns:
            for match in pattern.finditer(page_markdown):
                eq = match.group(1).strip()
                if eq:
                    all_equations.append(eq)

    doc.close()

    # Deduplicate while preserving order
    seen = set()
    unique_eq = []
    for eq in all_equations:
        if eq not in seen:
            seen.add(eq)
            unique_eq.append(eq)

    return unique_eq


def clean_section_text(text: str) -> str:
    """
    Remove footnote/footer noise embedded within section text.

    THE CORE PROBLEM THIS SOLVES:
    PyMuPDF joins ALL spans from a section with single spaces in segment_sections.
    Footer spans (conference stamps, arXiv lines, footnote markers) land in the
    middle of the joined string with no double-space boundary. Splitting on "  +"
    therefore can't isolate them — the entire string arrives as one fragment, and
    any keyword match wipes all content.

    THE FIX: Use regex substitution to remove noise as SUBSTRINGS in-place,
    then normalize whitespace. This preserves surrounding body text even when
    footer content is interleaved with it.
    """
    # Each pattern removes a specific class of footer/metadata fragment as a substring.
    # Ordered from most specific to least to avoid over-removal.
    inline_noise_subs = [
        # arXiv stamp: "arXiv:1703.06856v3 [stat.ML] 8 Mar 2018" (whole stamp)
        (re.compile(r'arXiv:\S+\s*\[[\w.]+\]\s*\d{1,2}\s+\w+\s+\d{4}', re.IGNORECASE), ' '),
        # arXiv stamp split across spans: "[stat.ML] 8 Mar 2018"
        (re.compile(r'\[\s*(stat|cs|math)\.[^\]]*\]\s*\d{1,2}\s+\w+\s+\d{4}', re.IGNORECASE), ' '),
        # Conference line: "31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA."
        (re.compile(r'\d+(st|nd|rd|th)\s+[A-Z][^.]*[Cc]onference[^.]*\.\s*'), ' '),
        # Generic conference/workshop/proceedings line up to next sentence
        (re.compile(r'[A-Z][^.]*(?:conference|workshop|proceedings)[^.]*\.\s*', re.IGNORECASE), ' '),
        # Footnote with URL: "2 https://obamawhitehouse...."
        (re.compile(r'\d+\s*https?://\S+\s*'), ' '),
        # Footnote equal contribution marker: "∗Equal contribution. This work was done..."
        (re.compile(r'[∗†‡§¶]\s*[Ee]qual contribution[^.]*\.\s*'), ' '),
        (re.compile(r'[∗†‡§¶]\s*[Tt]his work[^.]*\.\s*'), ' '),
        # Email addresses
        (re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+\s*'), ' '),
        # "Long Beach, CA, USA." as standalone footer fragment
        (re.compile(r'[Ll]ong [Bb]each[^.]*\.\s*'), ' '),
        # Lone page numbers at end of section (already handled by segment_sections but belt+suspenders)
        (re.compile(r'\s+\d{1,2}\s*$'), ' '),
    ]

    result = text
    for pattern, replacement in inline_noise_subs:
        result = pattern.sub(replacement, result)

    # Collapse multiple spaces and clean up
    result = re.sub(r'  +', ' ', result)
    result = result.strip()
    return result

def extract_title_from_page(pdf_path: str) -> str:
    """
    Extract paper title from first page using largest font size in top portion.

    Strategy: only consider spans in the top 40% of the page by y-coordinate.
    arXiv stamps, conference lines, and author blocks appear in the lower half.
    The title is almost always the largest text in the upper portion.

    Also filters out spans that look like metadata (arXiv IDs, dates, emails).
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_height = page.rect.height
    top_cutoff = page_height * 0.40  # only look in top 40% of page

    noise_title_patterns = [
        re.compile(r'arxiv', re.IGNORECASE),
        re.compile(r'\[stat\.|cs\.|math\.', re.IGNORECASE),
        re.compile(r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}', re.IGNORECASE),
        re.compile(r'@'),
        re.compile(r'^\d+$'),
    ]

    spans = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                y_pos = span["bbox"][1]
                if not text:
                    continue
                if y_pos > top_cutoff:          # skip anything below top 40%
                    continue
                if any(p.search(text) for p in noise_title_patterns):
                    continue
                spans.append((span["size"], y_pos, text))
    doc.close()

    if not spans:
        return ""

    max_size = max(s[0] for s in spans)
    title_spans = sorted(
        [s for s in spans if s[0] >= max_size - 0.5],
        key=lambda x: x[1]
    )
    title = " ".join(s[2] for s in title_spans).strip()
    return title if len(title) <= 200 else ""


def is_figure_caption(text: str) -> bool:
    """Detect a figure caption line: starts with 'Figure N:' or 'Fig. N:'"""
    return bool(re.match(r'^(figure|fig\.)\s*\d+\s*:', text.strip(), re.IGNORECASE))


def segment_sections(blocks: list[dict]) -> dict[str, str]:
    """
    Segment document into sections based on detected headings.

    KEY CHALLENGE — Figure/DAG debris:
    Academic PDFs embed causal graph figures whose node labels (A, X, Y, U)
    are individual 1-char PyMuPDF spans. Each span is too short to classify
    as debris individually (they look like variable names in body text).
    Only when joined do they form garbage like "A X Y U ( a ) ( b )...".

    SOLUTION — Figure sentinel buffering:
    When we encounter a figure caption span ("Figure 1: ..."), we know that
    ALL spans between the last real sentence end and this caption are figure
    node labels. We keep a pending_figure_buffer and only flush it when we
    confirm we've moved past the figure region (next real body sentence).
    This cleanly eliminates the DAG debris without affecting body text.
    """
    sections = {}
    current_section = "preamble"
    current_text = []
    # Buffer holds spans that MIGHT be figure debris (short/ambiguous spans
    # seen since the last confirmed body sentence)
    pending_buffer = []
    in_figure_zone = False  # True from figure caption until next body sentence

    _figure_caption_re = re.compile(r'^(figure|fig\.)\s*\d+\s*:', re.IGNORECASE)
    _body_sentence_re = re.compile(r'[a-zA-Z]{4,}.*[a-zA-Z]{4,}')  # has real words

    def is_real_body(text):
        """True if text looks like genuine body text (not a DAG label or short symbol)."""
        tokens = text.split()
        if len(tokens) < 3:
            return False
        # Must have at least one word longer than 4 chars
        long_words = sum(1 for t in tokens if len(t) > 4)
        return long_words >= 1

    for block in blocks:
        text = block["text"].strip()
        if not text:
            continue

        if block["is_heading"]:
            # Flush pending buffer — headings reset figure zone
            in_figure_zone = False
            pending_buffer = []
            if current_text:
                joined = " ".join(current_text).strip()
                joined = re.sub(r'\s+\d{1,2}\s*$', '', joined).strip()
                sections[current_section] = joined
            heading = text.lower().strip().split('\n')[0]
            heading = re.sub(r'^\d+[\.\ s]+', '', heading).strip()
            current_section = heading[:50]
            current_text = []

        elif _figure_caption_re.match(text):
            # This IS a figure caption. Everything in pending_buffer is DAG debris.
            # Discard the buffer and mark we're in the figure zone until next body.
            pending_buffer = []
            in_figure_zone = True
            # Also discard the caption itself — we don't want it in section content

        elif in_figure_zone:
            # We're past a figure caption. Only accept spans that look like real body.
            if is_real_body(text):
                in_figure_zone = False
                current_text.append(text)
            # else: still in figure zone (sub-captions, labels) — discard

        elif not is_real_body(text):
            # Ambiguous short span — could be a DAG label or a real symbol.
            # Buffer it; flush to current_text only if followed by real body text.
            pending_buffer.append(text)

        else:
            # Confirmed real body text. Flush pending buffer first, then add this.
            # (Buffered short spans before a real sentence are legitimate, e.g. math symbols)
            current_text.extend(pending_buffer)
            pending_buffer = []
            current_text.append(text)

    # Flush any remaining pending buffer and last section
    current_text.extend(pending_buffer)
    if current_text:
        joined = " ".join(current_text).strip()
        joined = re.sub(r'\s+\d{1,2}\s*$', '', joined).strip()
        sections[current_section] = joined

    return sections

def map_to_standard_keys(sections: dict) -> dict:
    """
    Map raw extracted section names to a comprehensive standard schema.

    Covers every major section type found across:
    - ML / Deep Learning / Computer Vision / NLP
    - Theoretical CS / Mathematics
    - Systems / HCI / Robotics
    - Biology / Medicine / Social Science (when adjacent to CS/ML)

    Output schema:
        abstract        Abstract / Synopsis
        introduction    Introduction / Contributions / Motivation
        related_work    Related Work / Literature Review / Prior Art
        background      Background / Preliminaries / Notation / Setup
        methodology     Methods / Model / Architecture / Theory / Algorithms
        datasets        Datasets / Corpus / Data Collection / Annotation
        experiments     Experiments / Setup / Baselines / Ablations / Evaluation
        results         Results / Performance / Findings / Accuracy
        discussion      Discussion / Analysis / Limitations / Broader Impact
        conclusion      Conclusion / Future Work / Summary
        equations       (populated separately by Nougat)
        metadata        (populated separately from PDF metadata)
        other           Supplemental, Appendix, References, unmatched

    Strategy:
      1. Supplemental/appendix prefixes intercepted first -> 'other'
      2. Keyword matching in priority order (specific before broad)
         First match wins; sections can accumulate (appended with separator)
      3. Large unmatched sections fall back into methodology
    """
    standard = {
        "abstract":     "",
        "introduction": "",
        "related_work": "",
        "background":   "",
        "methodology":  "",
        "datasets":     "",
        "experiments":  "",
        "results":      "",
        "discussion":   "",
        "conclusion":   "",
        "other":        {}
    }

    # ── Supplemental / appendix routing ───────────────────────────────────────
    # These section prefixes go directly to 'other', bypassing all keyword matching.
    # Prevents S1–S9 appendix sections from contaminating primary fields even when
    # their headings contain relevant keywords (e.g. "S4 Analysis of Pathways").
    supplemental_re = re.compile(
        r"^\s*("
        r"s\d+[\s\.\:\-]"               # S1, S2..., S1. S2:
        r"|appendix"                     # Appendix A, Appendix: ...
        r"|supplement"                   # Supplementary material
        r"|supp[\s\.]"                   # Supp., Supp A
        r"|acknowledgment"               # Acknowledgment(s)
        r"|acknowledgement"
        r"|reference"                    # References
        r"|bibliography"                 # Bibliography
        r"|funding"                      # Funding
        r"|conflict"                     # Conflict of interest
        r"|data availability"            # Data availability statement
        r"|code availability"            # Code/software availability
        r"|author contribution"          # Author contributions
        r"|competing interest"           # Competing interests
        r")",
        re.IGNORECASE
    )

    # ── Noise heading patterns ─────────────────────────────────────────────────
    # Section names matching these are discarded entirely (not even routed to other).
    # Their CONTENT is also discarded — so this is only for truly spurious headings
    # (page numbers, figure labels, arXiv stamps, and word-break artifacts).
    noise_patterns = [
        re.compile(r"arxiv",            re.IGNORECASE),
        re.compile(r"\d{4}\.\d{4,}"),   # arXiv IDs like 2301.12345
        re.compile(r"\[stat\.|cs\.|math\."),
        re.compile(r"^\d+\s*$"),         # lone page numbers
        re.compile(r"^(figure|table|fig\.)\s*\d", re.IGNORECASE),
        re.compile(r"^algorithm\s*\d",  re.IGNORECASE),
        re.compile(r"^equation\s*\d",   re.IGNORECASE),
        # Word-break artifacts: compound words split across lines by PyMuPDF
        # e.g. "counter|factual", "counter|factuals", "in|tuitively"
        re.compile(r"^factual$",        re.IGNORECASE),
        re.compile(r"^factuals$",       re.IGNORECASE),
        re.compile(r"^tional$",         re.IGNORECASE),   # "computa|tional"
        re.compile(r"^ational$",        re.IGNORECASE),   # "found|ational"
        re.compile(r"^ization$",        re.IGNORECASE),   # "normal|ization"
        re.compile(r"^isation$",        re.IGNORECASE),
    ]

    # ── Priority keyword map ────────────────────────────────────────────────────
    # Ordered top-to-bottom; FIRST match wins.
    # RULES:
    #   - More specific / narrower phrases listed BEFORE broader single words
    #   - Sections that could steal from each other are ordered carefully:
    #     conclusion > introduction > related_work > background > experiments
    #     > datasets > results > discussion > methodology
    #   - methodology is intentionally broadest and listed last
    key_map = [

        # ── Abstract ──────────────────────────────────────────────────────────
        ("abstract", [
            "abstract",
            "synopsis",
            "executive summary",
            "tl;dr",
        ]),

        # ── Conclusion ────────────────────────────────────────────────────────
        # Before introduction to prevent "summary" and "future work" grabbing intro.
        ("conclusion", [
            "conclusion", "conclusions",
            "concluding remarks", "concluding discussion",
            "final remarks", "final thoughts",
            "summary and conclusions", "summary and future work",
            "summary and outlook",
            "future work", "future directions", "future research",
            "limitations and future work", "limitations and conclusion",
            "broader impacts and conclusions",
            "closing remarks",
            "outlook",
        ]),

        # ── Introduction ──────────────────────────────────────────────────────
        ("introduction", [
            "introduction", "intro",
            "contribution", "contributions",
            "our contributions", "main contributions",
            "motivation", "motivating example",
            "overview",
            "problem statement", "problem definition", "problem formulation",
            "research question", "research questions",
            "research objective", "research objectives",
            "research goal", "research goals",
            "goals and contributions", "scope and contributions",
            "paper organization", "organization of the paper",
            "paper outline", "paper structure",
            "document organization",
        ]),

        # ── Related Work ──────────────────────────────────────────────────────
        ("related_work", [
            "related work", "related works",
            "prior work", "prior art",
            "literature review", "literature survey",
            "survey of related work",
            "background and related work", "related work and background",
            "related research",
            "previous work", "previous approaches",
            "prior approaches", "prior methods",
            "existing methods", "existing work", "existing approaches",
            "existing solutions", "existing techniques",
            "state of the art", "state-of-the-art",
            "comparison with related work",
            "connections to prior work",
            "related literature",
        ]),

        # ── Background / Preliminaries ────────────────────────────────────────
        # After related_work to avoid "background" stealing related work sections
        ("background", [
            "background",
            "preliminaries", "preliminary",
            "preliminary background",
            "notation and definitions", "notation and preliminaries",
            "definitions and notation",
            "mathematical notation", "mathematical background",
            "theoretical background", "theoretical foundation",
            "technical background", "technical preliminaries",
            "basic concepts", "basic background",
            "fundamental concepts",
            "problem setup", "formal setup",
            "formal background",
            "probabilistic background",
            "knowledge background",
            "structural equation", "structural causal",
            "counterfactual inference", "counterfactuals",
            "directed acyclic graph", "dag",
            "markov", "bayesian network",
            "probability", "probability theory",
            "random variable",
            "stochastic process",
            "statistical model",
            "linear model",
            "regression",
            "hypothesis",
            "set theory", "measure space",
        ]),

        # ── Experiments ───────────────────────────────────────────────────────
        # Before datasets to catch "evaluation data" etc. in experiment setup sections.
        # Before results to route "experimental results" here if section heading says so.
        ("experiments", [
            "experimental setup", "experimental settings", "experimental design",
            "experimental configuration", "experimental protocol",
            "experimental results",
            "experimental evaluation",
            "evaluation setup", "evaluation protocol",
            "evaluation metric", "evaluation metrics",
            "evaluation criteria",
            "experiment", "experiments", "experimental",
            "evaluation",
            "implementation detail", "implementation details",
            "implementation", "technical implementation",
            "computational setup", "hardware setup",
            "training detail", "training details",
            "hyperparameter", "hyperparameters",
            "hyperparameter setting", "hyperparameter tuning",
            "baseline", "baselines", "baseline methods",
            "baseline systems",
            "competing methods", "comparison methods",
            "ablation study", "ablation studies", "ablation",
            "human evaluation", "crowdsourcing", "user study",
            "annotation study",
            "simulation", "simulations",
            "illustration", "case study", "case studies",
            "proof of concept",
            "comparative study", "comparative evaluation", "comparison",
            "in-context learning experiments",
        ]),

        # ── Datasets ──────────────────────────────────────────────────────────
        # After experiments — precise dataset-specific terms only.
        # Avoids matching "data" as a bare substring in unrelated headings.
        ("datasets", [
            "dataset", "datasets",
            "data collection", "data gathering", "data acquisition",
            "data description", "dataset description",
            "dataset overview",
            "corpus", "corpora",
            "benchmark dataset", "benchmark suite",
            "data preprocessing", "preprocessing", "data preparation",
            "data pipeline",
            "data augmentation", "augmentation",
            "annotation", "annotations",
            "data annotation", "annotation process", "annotation guideline",
            "labeling", "data labeling",
            "data statistics", "dataset statistics",
            "data splits", "train/test split",
            "training data", "test data", "validation data",
        ]),

        # ── Results ───────────────────────────────────────────────────────────
        ("results", [
            "result", "results",
            "main result", "main results",
            "key result", "key results",
            "performance", "performance analysis", "performance comparison",
            "performance evaluation",
            "finding", "findings",
            "quantitative result", "quantitative results",
            "qualitative result", "qualitative results",
            "outcome", "outcomes",
            "observation", "observations",
            "accuracy", "accuracy analysis",
            "error analysis", "error breakdown",
            "benchmark result", "benchmark results",
            "leaderboard",
            "numerical result", "numerical results",
            "empirical result", "empirical results",
        ]),

        # ── Discussion ────────────────────────────────────────────────────────
        ("discussion", [
            "discussion",
            "further analysis", "additional analysis", "detailed analysis",
            "in-depth analysis",
            "qualitative analysis",
            "analysis and discussion",
            "interpretation", "interpretability", "model interpretation",
            "failure case", "failure cases", "failure analysis",
            "error analysis and discussion",
            "limitation", "limitations",
            "limitations and discussion",
            "broader impact", "societal impact", "social impact",
            "ethical consideration", "ethical considerations", "ethics",
            "potential risks", "risks and limitations",
            "threat to validity", "threats to validity",
            "impact statement",
        ]),

        # ── Methodology ───────────────────────────────────────────────────────
        # Broadest category — listed last among primary sections.
        # Catches all theoretical, algorithmic, design, and domain-specific content
        # that doesn't fit a more specific slot.
        ("methodology", [
            # Generic method terms
            "method", "methods", "methodology", "methodologies",
            "approach", "approaches",
            "proposed method", "proposed approach",
            "proposed framework", "proposed model",
            "our method", "our approach", "our framework", "our model",
            "our system", "our algorithm",
            # Architecture / system
            "model", "models",
            "architecture", "model architecture", "network architecture",
            "system", "system design", "system overview", "system description",
            "module", "modules", "component",
            # Algorithms / procedures
            "framework", "algorithm", "algorithms",
            "procedure", "procedures",
            "technique", "techniques",
            # Formalism / theory (CS/Math)
            "formulation", "problem formulation",
            "formal definition", "formal model", "formal framework",
            "definition", "definitions",
            "theorem", "theorems",
            "proof", "proofs", "proof sketch",
            "lemma", "lemmas", "corollary", "corollaries",
            "proposition", "propositions",
            "theory", "theoretical framework", "theoretical analysis",
            "complexity", "complexity analysis",
            "convergence", "convergence analysis",
            "bound", "bounds", "upper bound", "lower bound",
            "approximation", "approximation scheme",
            # Training / optimization
            "training", "training procedure", "training strategy",
            "optimization", "optimizer",
            "objective", "objective function",
            "loss", "loss function",
            "regularization",
            "fine-tuning", "finetuning",
            "pretraining", "pre-training",
            # Neural architecture specifics
            "network", "neural network",
            "encoder", "decoder", "encoder-decoder",
            "attention", "self-attention", "cross-attention",
            "transformer", "transformer model",
            "diffusion", "diffusion model",
            "flow", "normalizing flow",
            "convolution", "convolutional",
            "recurrent", "lstm", "gru",
            "generative", "generative model",
            "discriminative", "discriminator",
            "autoencoder", "vae", "gan",
            "reinforcement", "reward",
            "embedding", "embeddings",
            "tokenization", "tokenizer",
            "prompt", "prompting", "prompt design",
            "in-context learning",
            # Domain / field specific
            "fairness",
            "counterfactual",
            "causal",
            "inference",
            "probabilistic",
            "bayesian",
            "graph", "graph neural",
            "language model", "large language model",
            "multimodal",
            "vision",
            "speech",
            "knowledge",
            "reasoning",
            "planning",
            "object detection", "image classification",
            "semantic segmentation", "instance segmentation",
            "named entity recognition", "relation extraction",
            "question answering", "reading comprehension",
            "text classification", "sentiment analysis",
            "machine translation", "summarization",
            "speech recognition", "text-to-speech",
            "image generation", "image synthesis",
            "point cloud", "depth estimation",
            "pose estimation", "action recognition",
            "knowledge graph", "link prediction",
            "node classification", "graph classification",
            "recommendation", "retrieval",
            "clustering", "segmentation",
            "task", "tasks",
            "safety",
            "alignment",
            "uncertainty",
            "calibration",
            # Math paper specifics
            "construction", "existence", "uniqueness",
            "structure", "characterization",
            "classification", "invariant",
            "decomposition", "reduction",
            "representation", "representations",
            "space", "spaces",
            "operator", "operators",
            "mapping", "mappings",
            "measure", "measure theory",
            "topology", "topological",
            # Subsection names common in papers
            "designing",
            "implications",
            "example", "examples",
            "illustration",
            "further",
            "background",
        ]),
    ]

    # ── Main routing loop ──────────────────────────────────────────────────────
    for section_name, content in sections.items():
        name_lower = section_name.lower().strip()

        # 1. Discard noise headings
        if any(p.search(section_name) for p in noise_patterns):
            continue

        # 2. Route supplemental / appendix to other
        if supplemental_re.match(name_lower):
            content = clean_section_text(content)
            if content:
                standard["other"][section_name] = content
            continue

        # 3. Clean footer/footnote noise from content
        content = clean_section_text(content)
        if not content:
            continue

        # 4. Keyword match (priority order, first match wins)
        matched = False
        for std_key, keywords in key_map:
            if any(kw in name_lower for kw in keywords):
                if standard[std_key]:
                    standard[std_key] += "\n\n" + content
                else:
                    standard[std_key] = content
                matched = True
                break

        # 5. Fallback: absorb substantial unmatched content into methodology
        if not matched:
            if len(content) > 200 and not standard["methodology"]:
                standard["methodology"] = content
            else:
                standard["other"][section_name] = content

    return standard


def parse_paper(pdf_path: str) -> dict:
    """
    Main function: parse a research paper PDF into structured JSON.

    Output schema:
        abstract, introduction, related_work, background,
        methodology, datasets, experiments, results,
        discussion, conclusion,
        equations, metadata, other
    """
    print(f"[*] Parsing: {pdf_path}")

    # Step 1: Extract raw text blocks with metadata
    blocks = extract_text_blocks(pdf_path)
    print(f"    Extracted {len(blocks)} text spans")

    # Step 2: Detect headings
    annotated = detect_headings(blocks)

    # Step 3: Segment into sections
    sections = segment_sections(annotated)
    print(f"    Found {len(sections)} sections: {list(sections.keys())}")

    # Step 4: Map to standard 10-section structure
    structured = map_to_standard_keys(sections)

    # Step 5: Load Nougat once, extract equations
    processor, model, device = load_nougat()
    equations = extract_equations(pdf_path, processor, model, device)
    structured["equations"] = equations
    print(f"    Extracted {len(equations)} equations via Nougat")

    # Step 6: Metadata — embedded PDF data with fallback to largest-font text on page 1
    doc = fitz.open(pdf_path)
    embedded_title = doc.metadata.get("title", "").strip()
    page_count = doc.page_count
    embedded_author = doc.metadata.get("author", "")
    doc.close()

    structured["metadata"] = {
        "filename": Path(pdf_path).name,
        "pages": page_count,
        "title": embedded_title if embedded_title else extract_title_from_page(pdf_path),
        "author": embedded_author,
    }

    return structured


def main():
    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to_pdf> [output.json]")
        print("Example: python parser.py paper.pdf output.json")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_paper.json"

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    result = parse_paper(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    primary_keys = [
        "abstract", "introduction", "related_work", "background",
        "methodology", "datasets", "experiments", "results",
        "discussion", "conclusion"
    ]
    populated = [k for k in primary_keys if result.get(k)]
    empty     = [k for k in primary_keys if not result.get(k)]

    print(f"\n[✓] Done. Output saved to: {output_path}")
    print(f"    Populated ({len(populated)}): {populated}")
    print(f"    Empty     ({len(empty)}): {empty}")
    print(f"    Equations: {len(result['equations'])}")
    print(f"    Other sections: {list(result['other'].keys())}")


if __name__ == "__main__":
    main()