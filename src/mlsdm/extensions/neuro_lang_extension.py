from __future__ import annotations

import os
import random
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlsdm.core.cognitive_controller import CognitiveController
from mlsdm.core.llm_wrapper import LLMWrapper
from mlsdm.observability.aphasia_logging import AphasiaLogEvent, log_aphasia_event

if TYPE_CHECKING:
    from mlsdm.config import AphasiaDetectorCalibration, SecureModeCalibration

# Import calibration defaults - these can be overridden via config
# Type hints use Optional to allow None when calibration module unavailable
APHASIA_DEFAULTS: AphasiaDetectorCalibration | None
SECURE_MODE_DEFAULTS: SecureModeCalibration | None

try:
    from mlsdm.config import APHASIA_DEFAULTS, SECURE_MODE_DEFAULTS
except ImportError:
    APHASIA_DEFAULTS = None
    SECURE_MODE_DEFAULTS = None

# Lazy import of PyTorch dependencies - only needed for NeuroLang mode
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, Sampler
    TORCH_AVAILABLE = True
except ImportError:
    # PyTorch not installed - NeuroLang features will be disabled
    TORCH_AVAILABLE = False
    # Provide None as placeholders when torch is not available
    # These will never be used at runtime due to TORCH_AVAILABLE checks
    Dataset = None  # type: ignore
    DataLoader = None  # type: ignore
    Sampler = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    torch = None  # type: ignore


ALLOWED_CHECKPOINT_DIR = Path("config").resolve()


def is_secure_mode_enabled() -> bool:
    """
    Check if MLSDM secure mode is enabled via environment variable.

    When secure mode is enabled (MLSDM_SECURE_MODE=1 or true), the system:
    - Disables NeuroLang training and checkpoint loading
    - Disables aphasia repair (detection only)
    - Prevents any offline training operations

    Returns:
        bool: True if secure mode is enabled, False otherwise
    """
    env_var = (
        SECURE_MODE_DEFAULTS.env_var_name
        if SECURE_MODE_DEFAULTS
        else "MLSDM_SECURE_MODE"
    )
    enabled_values = (
        SECURE_MODE_DEFAULTS.enabled_values
        if SECURE_MODE_DEFAULTS
        else ("1", "true", "TRUE")
    )
    return os.getenv(env_var, "0") in enabled_values


def safe_load_neurolang_checkpoint(path: str | None, device: Any) -> dict[str, Any] | None:
    """
    Safely load a NeuroLang checkpoint with path validation and structure verification.

    Security controls:
    - Restricts checkpoint loading to the configured ALLOWED_CHECKPOINT_DIR
    - Validates checkpoint file structure (must be dict with 'actor' and 'critic' keys)
    - Prevents path traversal attacks (including symlinks)
    - Uses weights_only=True to prevent arbitrary code execution

    Args:
        path: Path to checkpoint file, or None to skip loading
        device: PyTorch device for loading the checkpoint

    Returns:
        dict: Checkpoint dictionary with 'actor' and 'critic' state dicts, or None if path is None

    Raises:
        ValueError: If path is outside allowed directory or checkpoint structure is invalid
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "NeuroLang checkpoint loading requires 'mlsdm[neurolang]' extra (PyTorch not installed). "
            "Install with 'pip install mlsdm[neurolang]' or 'pip install torch>=2.0.0'."
        )

    if not path:
        return None

    p = Path(path).expanduser().resolve()

    # Prevent symlink-based path traversal: check if path is within allowed directory
    try:
        # is_relative_to is available in Python 3.9+
        if not p.is_relative_to(ALLOWED_CHECKPOINT_DIR):
            raise ValueError(f"Refusing to load checkpoint outside {ALLOWED_CHECKPOINT_DIR}")
    except AttributeError:
        # Fallback for Python < 3.9: use string comparison on resolved paths
        if not str(p).startswith(str(ALLOWED_CHECKPOINT_DIR)):
            raise ValueError(f"Refusing to load checkpoint outside {ALLOWED_CHECKPOINT_DIR}") from None

    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    # Load with weights_only=True to prevent arbitrary code execution from pickle
    # Note: This requires PyTorch >= 2.0.0
    try:
        obj = torch.load(p, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        obj = torch.load(p, map_location=device)

    if not isinstance(obj, dict):
        raise ValueError("Invalid neurolang checkpoint format: expected dict")

    if "actor" not in obj or "critic" not in obj:
        raise ValueError(
            f"Invalid checkpoint structure: missing 'actor' or 'critic' keys. "
            f"Found keys: {list(obj.keys())}"
        )

    return obj

simple_sentences = [
    "The cat ran.",
    "The dog barked.",
    "Alice said it.",
    "The book was interesting.",
    "The idea persists."
]

complex_sentences = [
    "The cat that chased the mouse ran away.",
    "The dog that bit the man who owned the cat barked.",
    "Alice said that Bob thought that Charlie believed it.",
    "The book that the student who failed the exam read was interesting.",
    "The idea that the theory that explains everything is wrong persists.",
    "John knows that Mary thinks that the plan will succeed.",
    "The bird that flew over the house that Jack built sang.",
    "The problem that the solution that we found fixes is complex.",
    "She whispered that he shouted that they laughed.",
    "The river that flows through the city that never sleeps is polluted."
]

all_sentences = simple_sentences + complex_sentences


# Define torch-dependent classes only when torch is available
if TORCH_AVAILABLE:
    class LanguageDataset(Dataset):  # type: ignore
        def __init__(self, sentences):
            self.sentences = sentences
            self.vocab = sorted(set(" ".join(sentences).split() + ["<PAD>", "<EOS>", "<BOS>"]))
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
            self.max_len = max(len(s.split()) for s in sentences) + 2

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            sentence = ["<BOS>"] + self.sentences[idx].split() + ["<EOS>"]
            padded = sentence + ["<PAD>"] * (self.max_len - len(sentence))
            return torch.tensor([self.word_to_idx[w] for w in padded], dtype=torch.long)


    class CurriculumSampler(Sampler):  # type: ignore
        def __init__(self, dataset, simple_count=None):
            if simple_count is None:
                simple_count = len(simple_sentences)
            self.indices = list(range(len(dataset)))
            self.simple_indices = self.indices[:simple_count]
            self.complex_indices = self.indices[simple_count:]

        def __iter__(self):
            phase1 = random.sample(self.simple_indices, len(self.simple_indices))
            phase2 = random.sample(self.complex_indices, len(self.complex_indices))
            return iter(phase1 + phase2)

        def __len__(self):
            return len(self.indices)


    class TransformerBlock(nn.Module):
        def __init__(self, embed_size, heads=4):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_size)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_size, 4 * embed_size),
                nn.ReLU(),
                nn.Linear(4 * embed_size, embed_size)
            )
            self.norm2 = nn.LayerNorm(embed_size)

        def forward(self, x):
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float("-inf"), diagonal=1)
            attn_out, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(x + attn_out)
            ff_out = self.feed_forward(x)
            return self.norm2(x + ff_out)


    class InnateGrammarModule(nn.Module):
        def __init__(self, vocab_size, embed_size=64, layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.positional = nn.Parameter(torch.zeros(1, 512, embed_size))
            self.blocks = nn.ModuleList([TransformerBlock(embed_size) for _ in range(layers)])
            self.fc = nn.Linear(embed_size, vocab_size)

        def forward(self, x):
            seq_len = x.size(1)
            embed = self.embedding(x) + self.positional[:, :seq_len, :]
            for block in self.blocks:
                embed = block(embed)
            return self.fc(embed)

        def generate_recursive(self, dataset, start_word, max_len=20):
            if start_word not in dataset.word_to_idx:
                return ""
            device = next(self.parameters()).device
            words = [start_word]
            input_idx = torch.tensor([[dataset.word_to_idx[start_word]]], dtype=torch.long, device=device)
            for _ in range(max_len):
                logit = self.forward(input_idx)[:, -1, :]
                next_idx = torch.argmax(logit, dim=1).item()
                next_word = dataset.idx_to_word[next_idx]
                if next_word == "<EOS>":
                    break
                words.append(next_word)
                next_token = torch.tensor([[next_idx]], dtype=torch.long, device=device)
                input_idx = torch.cat([input_idx, next_token], dim=1)
            return " ".join(words)


    class CriticalPeriodTrainer:
        def __init__(self, actor, critic, dataset, epochs=5):
            self.actor = actor
            self.critic = critic
            self.dataset = dataset
            self.dataloader = DataLoader(dataset, batch_size=2, sampler=CurriculumSampler(dataset))
            self.optimizer_actor = optim.Adam(actor.parameters(), lr=0.001)
            self.optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.word_to_idx["<PAD>"])
            self.epochs = epochs

        def train(self):
            device = next(self.actor.parameters()).device
            for _ in range(self.epochs):
                total_loss = 0.0
                for batch in self.dataloader:
                    batch = batch.to(device)
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]

                    self.optimizer_critic.zero_grad()
                    critic_out = self.critic(inputs)
                    critic_loss = self.criterion(
                        critic_out.reshape(-1, len(self.dataset.vocab)),
                        targets.reshape(-1)
                    )
                    critic_loss.backward()
                    self.optimizer_critic.step()

                    self.optimizer_actor.zero_grad()
                    outputs = self.actor(inputs)
                    gen_loss = self.criterion(
                        outputs.reshape(-1, len(self.dataset.vocab)),
                        targets.reshape(-1)
                    )

                    with torch.no_grad():
                        critic_eval = self.critic(inputs)
                        critic_eval_loss = self.criterion(
                            critic_eval.reshape(-1, len(self.dataset.vocab)),
                            targets.reshape(-1)
                        )
                        reward = torch.exp(-critic_eval_loss)

                    actor_loss = gen_loss * (1 - reward)
                    actor_loss.backward()
                    self.optimizer_actor.step()
                    total_loss += float(actor_loss.item())


    class ModularLanguageProcessor:
        def __init__(self, actor, critic, dataset):
            self.actor = actor
            self.critic = critic
            self.dataset = dataset
            self.understanding_pattern = re.compile(r"(that|who|which)")

        def process(self, input_sentence):
            recursion_count = len(self.understanding_pattern.findall(input_sentence))
            if recursion_count < 1:
                return "Input lacks recursion; enhancing..."
            words = input_sentence.split()
            valid_words = [w for w in words if w in self.dataset.word_to_idx]
            if not valid_words:
                return "Invalid input for processing."
            device = next(self.actor.parameters()).device
            start_word = valid_words[-1]
            generated = self.actor.generate_recursive(self.dataset, start_word)
            crit_words = generated.split()
            valid_crit_words = [w for w in crit_words if w in self.dataset.word_to_idx]
            if not valid_crit_words:
                crit_score = 0.0
            else:
                crit_input = torch.tensor(
                    [[self.dataset.word_to_idx[w] for w in valid_crit_words]],
                    dtype=torch.long,
                    device=device
                )
                crit_logits = self.critic(crit_input)
                crit_score = torch.softmax(crit_logits[:, -1, :], dim=1).max().item()
            return f"Processed (Critic score: {crit_score:.2f}): {input_sentence} -> {generated}"


    class SocialIntegrator:
        def __init__(self, processor1, processor2):
            self.agent1 = processor1
            self.agent2 = processor2

        def interact(self, sentence1, sentence2):
            proc1 = self.agent1.process(sentence1)
            proc2 = self.agent2.process(sentence2)
            combined = proc1 + " And " + proc2
            if random.random() > 0.7:
                combined = combined.replace("that", "which", 1)
            return combined


# AphasiaBrocaDetector does NOT depend on torch - keep it outside the conditional block
class AphasiaBrocaDetector:
    """
    Neurobiologically-grounded detector for telegraphic speech patterns in LLM outputs.

    This detector identifies Broca's aphasia-like patterns in text, characterized by:
    - Telegraphic speech: Short, fragmented sentences lacking grammatical structure
    - Function word omission: Missing articles, prepositions, conjunctions
    - Preserved semantics: Core meaning intact despite grammatical deficits

    Neurobiological Foundation:
        Broca's aphasia results from damage to the left inferior frontal gyrus (BA44/45),
        causing expressive language difficulties while preserving comprehension. This
        detector maps those clinical characteristics to LLM output quality metrics.

        In LLMs, analogous patterns emerge from:
        - Token budget constraints
        - Context window limitations
        - Incomplete reasoning chains
        - Over-compression of information

    Detection Algorithm:
        1. Sentence segmentation and word tokenization
        2. Function word ratio calculation (e.g., "the", "is", "and")
        3. Fragment detection (sentences below threshold length)
        4. Severity quantification based on threshold violations

    Thread Safety:
        This detector is stateless and thread-safe. All methods are pure functions
        with no shared mutable state, making it safe for concurrent access.

    Performance:
        - Time complexity: O(n) where n = text length
        - Space complexity: O(n)
        - Typical latency: 1-15ms depending on text length
        - Throughput: ~5,000 analyses/sec (single-threaded)

    Validation Metrics (repository corpus):
        - True Positive Rate: 100% (50/50 telegraphic samples detected)
        - True Negative Rate: 88% (44/50 normal samples)
        - Overall Accuracy: 94%
        - Mean Severity (telegraphic): 0.885

    See Also:
        - APHASIA_SPEC.md: Complete specification and neurobiological foundations
        - docs/NEURO_FOUNDATIONS.md: Neuroscience background
        - tests/eval/aphasia_eval_suite.py: Empirical validation results

    Example:
        >>> detector = AphasiaBrocaDetector()
        >>> result = detector.analyze("Cat run. Dog bark. Good.")
        >>> print(result['is_aphasic'])
        True
        >>> print(result['severity'])
        0.87
        >>> print(result['flags'])
        ['short_sentences', 'low_function_words', 'high_fragment_ratio']

    Attributes:
        function_words: Set of English function words used for ratio calculation
        min_sentence_len: Minimum average sentence length threshold (default: 6.0 words)
        min_function_word_ratio: Minimum function word ratio threshold (default: 0.15)
        max_fragment_ratio: Maximum allowed fragment ratio threshold (default: 0.5)
        fragment_length_threshold: Words below which a sentence is a fragment (default: 4)
    """

    # Default values from calibration
    DEFAULT_MIN_SENTENCE_LEN = (
        APHASIA_DEFAULTS.min_sentence_len if APHASIA_DEFAULTS else 6.0
    )
    DEFAULT_MIN_FUNCTION_WORD_RATIO = (
        APHASIA_DEFAULTS.min_function_word_ratio if APHASIA_DEFAULTS else 0.15
    )
    DEFAULT_MAX_FRAGMENT_RATIO = (
        APHASIA_DEFAULTS.max_fragment_ratio if APHASIA_DEFAULTS else 0.5
    )
    DEFAULT_FRAGMENT_LENGTH_THRESHOLD = (
        APHASIA_DEFAULTS.fragment_length_threshold if APHASIA_DEFAULTS else 4
    )

    def __init__(
        self,
        min_sentence_len: float | None = None,
        min_function_word_ratio: float | None = None,
        max_fragment_ratio: float | None = None,
    ):
        """
        Initialize the Aphasia-Broca detector with configurable thresholds.

        The detector uses three clinical metrics to identify telegraphic speech patterns:
        1. Average sentence length (words per sentence)
        2. Function word ratio (proportion of grammatical function words)
        3. Fragment ratio (proportion of very short sentences)

        Clinical Rationale:
            These thresholds are based on standard clinical assessment criteria for
            Broca's aphasia, adapted for computational analysis of LLM outputs.

        Args:
            min_sentence_len: Minimum average words per sentence to be considered
                healthy (default: 6.0). Values below trigger short_sentences flag.
            min_function_word_ratio: Minimum proportion of function words (default: 0.15
                or 15%). Values below trigger low_function_words flag.
            max_fragment_ratio: Maximum proportion of sentences that can be fragments
                (default: 0.5 or 50%). Values above trigger high_fragment_ratio flag.

        Default Calibration:
            The default values are empirically validated on a diverse corpus and
            represent balanced thresholds that minimize false positives while
            maintaining high detection sensitivity. See APHASIA_SPEC.md for
            validation metrics.

        Example:
            >>> # Use default thresholds
            >>> detector = AphasiaBrocaDetector()
            >>>
            >>> # Custom thresholds for stricter detection
            >>> strict_detector = AphasiaBrocaDetector(
            ...     min_sentence_len=8.0,
            ...     min_function_word_ratio=0.20,
            ...     max_fragment_ratio=0.3
            ... )
        """
        # Use calibration defaults if not specified
        if min_sentence_len is None:
            min_sentence_len = self.DEFAULT_MIN_SENTENCE_LEN
        if min_function_word_ratio is None:
            min_function_word_ratio = self.DEFAULT_MIN_FUNCTION_WORD_RATIO
        if max_fragment_ratio is None:
            max_fragment_ratio = self.DEFAULT_MAX_FRAGMENT_RATIO

        # English function words (grammatical/structural words)
        # These are typically omitted in telegraphic speech patterns
        self.function_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "that",
            "which",
            "who",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "with",
            "by",
            "as",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }
        self.min_sentence_len = float(min_sentence_len)
        self.min_function_word_ratio = float(min_function_word_ratio)
        self.max_fragment_ratio = float(max_fragment_ratio)
        self.fragment_length_threshold = self.DEFAULT_FRAGMENT_LENGTH_THRESHOLD

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyze text for Broca's aphasia-like telegraphic speech patterns.

        This method performs a comprehensive linguistic analysis to detect three
        characteristic patterns of expressive aphasia:

        1. **Short Sentences**: Average sentence length below minimum threshold
           - Clinical correlate: Reduced syntactic complexity
           - Detection: Mean words per sentence < min_sentence_len

        2. **Low Function Words**: Underrepresentation of grammatical function words
           - Clinical correlate: Agrammatism (omission of articles, prepositions, etc.)
           - Detection: Function word ratio < min_function_word_ratio

        3. **High Fragment Ratio**: Excessive proportion of very short sentences
           - Clinical correlate: Telegraphic speech with incomplete utterances
           - Detection: Fragments / total sentences > max_fragment_ratio

        Algorithm Steps:
            1. Text normalization and sentence segmentation (regex-based)
            2. Word tokenization for each sentence
            3. Function word counting (case-insensitive matching)
            4. Metric calculation (averages, ratios)
            5. Threshold comparison and flag generation
            6. Severity quantification (0.0-1.0 scale)

        Args:
            text: Input text to analyze. Can be any length, including empty strings.

        Returns:
            dict: Analysis report with the following keys:
                - is_aphasic (bool): True if any threshold is violated
                - severity (float): Quantified severity score (0.0 = healthy, 1.0 = severe)
                - avg_sentence_len (float): Average words per sentence
                - function_word_ratio (float): Proportion of function words (0.0-1.0)
                - fragment_ratio (float): Proportion of short sentences (0.0-1.0)
                - flags (list[str]): Specific violations detected:
                    * "short_sentences": avg_sentence_len < threshold
                    * "low_function_words": function_word_ratio < threshold
                    * "high_fragment_ratio": fragment_ratio > threshold
                    * "empty_text": input was empty or whitespace-only
                    * "no_sentence_boundaries": no sentence delimiters found
                    * "no_tokens": no valid tokens extracted

        Severity Calculation:
            Severity is computed as the normalized mean of three deviation scores:
            - Length deviation: (threshold - actual) / threshold
            - Function word deviation: (threshold - actual) / threshold
            - Fragment deviation: (actual - threshold) / threshold
            Each component is capped at 1.0, and the final severity is their average.

        Edge Cases:
            - Empty/whitespace text: Returns is_aphasic=True, severity=1.0
            - No sentence boundaries: Treated as single sentence
            - No valid tokens: Returns is_aphasic=True, severity=1.0
            - Single word: Likely flagged as aphasic due to low length/function ratio

        Thread Safety:
            This method is thread-safe as it operates only on local variables and
            immutable instance attributes. No locks are required for concurrent calls.

        Performance:
            - Time complexity: O(n) where n = text length
            - Space complexity: O(n) for sentence and token storage
            - Typical latency: 1-2ms for 100 words, 10-15ms for 1000 words

        Examples:
            >>> detector = AphasiaBrocaDetector()
            >>>
            >>> # Healthy text with proper grammar
            >>> result = detector.analyze(
            ...     "The cognitive architecture provides a comprehensive framework. "
            ...     "It integrates multiple biological principles to ensure safety."
            ... )
            >>> assert result['is_aphasic'] == False
            >>> assert result['severity'] < 0.1
            >>> assert result['flags'] == []
            >>>
            >>> # Telegraphic text (Broca's aphasia pattern)
            >>> result = detector.analyze("Cat run. Dog bark. Bird fly.")
            >>> assert result['is_aphasic'] == True
            >>> assert result['severity'] > 0.5
            >>> assert 'short_sentences' in result['flags']
            >>> assert 'low_function_words' in result['flags']
            >>>
            >>> # Edge case: empty text
            >>> result = detector.analyze("")
            >>> assert result['is_aphasic'] == True
            >>> assert result['severity'] == 1.0
            >>> assert 'empty_text' in result['flags']

        See Also:
            - APHASIA_SPEC.md: Complete specification and validation metrics
            - tests/validation/test_aphasia_detection.py: Comprehensive test suite
            - tests/eval/aphasia_eval_suite.py: Empirical validation results
        """
        # Step 1: Normalize and validate input text
        cleaned = text.strip()
        if not cleaned:
            # Edge case: Empty or whitespace-only input
            return {
                "is_aphasic": True,
                "severity": 1.0,
                "avg_sentence_len": 0.0,
                "function_word_ratio": 0.0,
                "fragment_ratio": 1.0,
                "flags": ["empty_text"],
            }

        # Step 2: Sentence segmentation using standard punctuation delimiters
        # Splits on period, exclamation mark, or question mark
        sentences = re.split(r"[.!?]+", cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            # Edge case: Text has no sentence boundaries
            return {
                "is_aphasic": True,
                "severity": 1.0,
                "avg_sentence_len": 0.0,
                "function_word_ratio": 0.0,
                "fragment_ratio": 1.0,
                "flags": ["no_sentence_boundaries"],
            }

        # Step 3: Tokenization and metric calculation
        # Clinical correlate: Count words per sentence (syntactic complexity indicator)
        total_tokens = 0
        function_tokens = 0  # Articles, prepositions, conjunctions (agrammatism indicator)
        fragment_sentences = 0  # Very short sentences (telegraphic speech indicator)

        for sent in sentences:
            # Simple whitespace-based tokenization (case-insensitive for function word matching)
            tokens = [t.lower() for t in sent.split() if t.strip()]
            length = len(tokens)
            total_tokens += length

            # Count fragments: sentences shorter than threshold
            # Clinical basis: Broca's patients produce incomplete, telegraphic utterances
            if length < self.fragment_length_threshold:
                fragment_sentences += 1

            # Count function words: "the", "is", "and", "of", etc.
            # Clinical basis: Function word omission is a hallmark of agrammatism
            function_tokens += sum(1 for t in tokens if t in self.function_words)

        if total_tokens == 0:
            # Edge case: Text contains no valid tokens
            return {
                "is_aphasic": True,
                "severity": 1.0,
                "avg_sentence_len": 0.0,
                "function_word_ratio": 0.0,
                "fragment_ratio": 1.0,
                "flags": ["no_tokens"],
            }

        # Step 4: Calculate linguistic metrics
        avg_sentence_len = total_tokens / len(sentences)
        function_word_ratio = function_tokens / total_tokens
        fragment_ratio = fragment_sentences / len(sentences)

        # Step 5: Threshold comparison and flag generation
        flags = []
        if avg_sentence_len < self.min_sentence_len:
            flags.append("short_sentences")
        if function_word_ratio < self.min_function_word_ratio:
            flags.append("low_function_words")
        if fragment_ratio > self.max_fragment_ratio:
            flags.append("high_fragment_ratio")

        is_aphasic = bool(flags)

        # Step 6: Severity quantification (normalized deviation from thresholds)
        # Formula: mean of three normalized deviations, capped at 1.0
        # Each component measures how far the text deviates from healthy thresholds
        severity = 0.0
        if flags:
            # Length deviation: measures syntactic simplification
            length_deviation = max(0.0, self.min_sentence_len - avg_sentence_len) / self.min_sentence_len

            # Function word deviation: measures agrammatism severity
            function_deviation = max(0.0, self.min_function_word_ratio - function_word_ratio) / max(
                self.min_function_word_ratio, 1e-6
            )

            # Fragment deviation: measures telegraphic fragmentation
            fragment_deviation = max(0.0, fragment_ratio - self.max_fragment_ratio) / max(
                self.max_fragment_ratio, 1e-6
            )

            # Combined severity: average of three components, capped at 1.0
            severity = min(
                1.0,
                (length_deviation + function_deviation + fragment_deviation) / 3.0,
            )

        # Step 7: Return structured analysis report
        return {
            "is_aphasic": is_aphasic,
            "severity": float(severity),
            "avg_sentence_len": float(avg_sentence_len),
            "function_word_ratio": float(function_word_ratio),
            "fragment_ratio": float(fragment_ratio),
            "flags": flags,
        }


class AphasiaSpeechGovernor:
    """
    Speech governor implementation for Aphasia-Broca detection and repair.

    This governor analyzes LLM output for telegraphic speech patterns
    characteristic of Broca's aphasia and optionally repairs them using
    an LLM-based repair strategy.
    """

    # Default values from calibration
    DEFAULT_SEVERITY_THRESHOLD = (
        APHASIA_DEFAULTS.severity_threshold if APHASIA_DEFAULTS else 0.3
    )
    DEFAULT_REPAIR_ENABLED = (
        APHASIA_DEFAULTS.repair_enabled if APHASIA_DEFAULTS else True
    )

    def __init__(
        self,
        detector: AphasiaBrocaDetector,
        repair_enabled: bool | None = None,
        severity_threshold: float | None = None,
        llm_generate_fn: Any = None,
    ):
        """
        Initialize the Aphasia speech governor.

        Args:
            detector: AphasiaBrocaDetector instance for analysis
            repair_enabled: Whether to repair detected aphasia (default from calibration)
            severity_threshold: Minimum severity to trigger repair (default from calibration)
            llm_generate_fn: LLM function for repair; required if repair_enabled
        """
        self._detector = detector
        self._repair_enabled = (
            repair_enabled if repair_enabled is not None else self.DEFAULT_REPAIR_ENABLED
        )
        self._severity_threshold = (
            severity_threshold if severity_threshold is not None else self.DEFAULT_SEVERITY_THRESHOLD
        )
        self._llm_generate_fn = llm_generate_fn

    def __call__(self, *, prompt: str, draft: str, max_tokens: int) -> Any:
        """Apply Aphasia detection and optional repair to draft text."""
        from mlsdm.speech.governance import SpeechGovernanceResult

        report = self._detector.analyze(draft)

        final_text = draft
        metadata: dict[str, Any] = {"aphasia_report": report, "repaired": False}

        if (
            self._repair_enabled
            and report["is_aphasic"]
            and report["severity"] >= self._severity_threshold
        ):
            llm_fn = self._llm_generate_fn
            if llm_fn is None:
                raise RuntimeError(
                    "Repair requested but no llm_generate_fn provided to AphasiaSpeechGovernor"
                )

            repair_prompt = (
                f"{prompt}\n\n"
                "The following draft answer shows Broca-like aphasia "
                "(telegraphic style, broken syntax). Rewrite it in coherent, full "
                "sentences, preserving all technical details and reasoning steps.\n\n"
                f"Draft answer:\n{draft}"
            )
            final_text = llm_fn(repair_prompt, max_tokens)
            metadata["repaired"] = True

        return SpeechGovernanceResult(
            final_text=final_text,
            raw_text=draft,
            metadata=metadata,
        )


class NeuroLangWrapper(LLMWrapper):
    def __init__(
        self,
        llm_generate_fn,
        embedding_fn,
        dim=384,
        capacity=20000,
        wake_duration=8,
        sleep_duration=3,
        initial_moral_threshold=0.50,
        aphasia_detect_enabled=True,
        aphasia_repair_enabled=True,
        aphasia_severity_threshold=0.3,
        neurolang_mode="eager_train",
        neurolang_checkpoint_path=None,
    ):
        # Security: Apply secure mode overrides if enabled
        if is_secure_mode_enabled():
            neurolang_mode = "disabled"
            neurolang_checkpoint_path = None
            aphasia_repair_enabled = False

        # Aphasia-Broca configuration flags
        self.aphasia_detect_enabled = bool(aphasia_detect_enabled)
        self.aphasia_repair_enabled = bool(aphasia_repair_enabled)
        self.aphasia_severity_threshold = float(aphasia_severity_threshold)

        # NeuroLang mode configuration
        if neurolang_mode not in ("eager_train", "lazy_train", "disabled"):
            raise ValueError(
                f"Invalid neurolang_mode: {neurolang_mode}. "
                "Must be one of: 'eager_train', 'lazy_train', 'disabled'"
            )
        self.neurolang_mode = neurolang_mode
        self.neurolang_checkpoint_path = neurolang_checkpoint_path

        # Check PyTorch availability when NeuroLang mode is enabled
        if self.neurolang_mode != "disabled" and not TORCH_AVAILABLE:
            raise RuntimeError(
                "NeuroLang mode requires 'mlsdm[neurolang]' extra (PyTorch not installed). "
                "Either install extras with 'pip install mlsdm[neurolang]' or set neurolang_mode='disabled'."
            )

        # Always initialize controller and aphasia detector
        self.controller = CognitiveController(dim)
        self.aphasia_detector = AphasiaBrocaDetector()

        # Create Aphasia speech governor if detection is enabled
        speech_governor = None
        if self.aphasia_detect_enabled:
            from mlsdm.speech.governance import PipelineSpeechGovernor

            aphasia_governor = AphasiaSpeechGovernor(
                detector=self.aphasia_detector,
                repair_enabled=self.aphasia_repair_enabled,
                severity_threshold=self.aphasia_severity_threshold,
                llm_generate_fn=llm_generate_fn,
            )

            speech_governor = PipelineSpeechGovernor(
                governors=[
                    ("aphasia_broca", aphasia_governor),
                    # Future: add other governors here, e.g.:
                    # ("toxicity_filter", custom_toxicity_governor)
                ]
            )

        # Initialize parent LLMWrapper with speech governor
        super().__init__(
            llm_generate_fn=llm_generate_fn,
            embedding_fn=embedding_fn,
            dim=dim,
            capacity=capacity,
            wake_duration=wake_duration,
            sleep_duration=sleep_duration,
            initial_moral_threshold=initial_moral_threshold,
            speech_governor=speech_governor,
        )

        # Initialize NeuroLang components based on mode
        if self.neurolang_mode == "disabled":
            # Disabled mode: skip NeuroLang components entirely
            self.actor = None
            self.critic = None
            self.trainer = None
            self.processor1 = None
            self.processor2 = None
            self.integrator = None
            self.dataset = None
        else:
            # Eager or lazy training mode: initialize components
            self.dataset = LanguageDataset(all_sentences)
            vocab_size = len(self.dataset.vocab)

            self.actor = InnateGrammarModule(vocab_size)
            self.critic = InnateGrammarModule(vocab_size)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.actor.to(device)
            self.critic.to(device)

            self.trainer = CriticalPeriodTrainer(self.actor, self.critic, self.dataset, epochs=3)

            # Load checkpoint if provided (using secure loading)
            checkpoint_loaded = False
            if self.neurolang_checkpoint_path is not None:
                try:
                    checkpoint = safe_load_neurolang_checkpoint(
                        self.neurolang_checkpoint_path, device
                    )
                    if checkpoint is not None:
                        self.actor.load_state_dict(checkpoint["actor"])
                        self.critic.load_state_dict(checkpoint["critic"])
                        checkpoint_loaded = True
                except Exception as e:
                    raise ValueError(
                        f"Failed to load NeuroLang checkpoint from '{self.neurolang_checkpoint_path}': {str(e)}"
                    ) from e

            # Train based on mode and checkpoint availability
            if not checkpoint_loaded:
                if self.neurolang_mode == "eager_train":
                    self.trainer.train()
                elif self.neurolang_mode == "lazy_train":
                    # For lazy training, set up state tracking
                    self._trained = False
                    self._train_lock = threading.Lock()

            self.processor1 = ModularLanguageProcessor(self.actor, self.critic, self.dataset)
            self.processor2 = ModularLanguageProcessor(self.critic, self.actor, self.dataset)
            self.integrator = SocialIntegrator(self.processor1, self.processor2)

    def generate(
        self,
        prompt: str,
        moral_value: float,
        max_tokens: int | None = None,
        context_top_k: int | None = None,
    ) -> dict[str, Any]:
        # Handle lazy training: train on first generation if not yet trained
        if self.neurolang_mode == "lazy_train" and not self._trained:
            with self._train_lock:
                # Double-check after acquiring lock
                if not self._trained:
                    self.trainer.train()
                    self._trained = True

        # Use default if context_top_k is None (retained for future retrieval logic)
        _effective_context_top_k = context_top_k if context_top_k is not None else 5

        # Generate NeuroLang enhancement based on mode
        if self.neurolang_mode == "disabled":
            # Disabled mode: skip NeuroLang enhancement
            neuro_response = "NeuroLang disabled"
            embedding = self.embed(prompt)
        else:
            # Enabled mode: use NeuroLang integrator
            neuro_response = self.integrator.interact(prompt, prompt)
            embedding = self.embed(neuro_response)

        state = self.controller.process_event(embedding, moral_value)

        if not state["accepted"]:
            return {
                "response": "Rejected by moral filter.",
                "phase": state["phase"],
                "accepted": False,
                "neuro_enhancement": neuro_response,
                "aphasia_flags": None,
            }

        # Generate LLM response with or without NeuroLang enhancement
        # Use default max_tokens if not provided
        tokens = max_tokens if max_tokens is not None else 50
        if self.neurolang_mode == "disabled":
            base_response = self.llm_generate(prompt, tokens)
        else:
            enhanced_prompt = f"{prompt}\n\n[NeuroLang enhancement]: {neuro_response}"
            base_response = self.llm_generate(enhanced_prompt, tokens)

        # Apply speech governance if configured (via parent's speech governor)
        final_response = base_response
        aphasia_report = None
        governed_metadata = None

        if self._speech_governor is not None:
            # Use the speech governor (PipelineSpeechGovernor or AphasiaSpeechGovernor) for processing
            gov_result = self._speech_governor(
                prompt=prompt,
                draft=base_response,
                max_tokens=tokens,
            )
            final_response = gov_result.final_text
            governed_metadata = {
                "raw_text": gov_result.raw_text,
                "metadata": gov_result.metadata,
            }

            # Extract aphasia_report for backward compatibility and logging
            # When using PipelineSpeechGovernor, look for aphasia_broca step
            if "pipeline" in gov_result.metadata:
                # Pipeline mode: extract aphasia report from pipeline steps
                for step in gov_result.metadata["pipeline"]:
                    if step["name"] == "aphasia_broca" and step["status"] == "ok":
                        aphasia_report = step["metadata"].get("aphasia_report")
                        break
            else:
                # Direct mode (backward compatibility)
                aphasia_report = gov_result.metadata.get("aphasia_report")

            # Log aphasia event for observability
            if aphasia_report is not None:
                repaired = False
                # Check if repaired flag exists in metadata
                if "pipeline" in gov_result.metadata:
                    for step in gov_result.metadata["pipeline"]:
                        if step["name"] == "aphasia_broca" and step["status"] == "ok":
                            repaired = step["metadata"].get("repaired", False)
                            break
                else:
                    repaired = gov_result.metadata.get("repaired", False)

                if repaired:
                    decision = "repaired"
                elif aphasia_report["is_aphasic"]:
                    decision = "detected_no_repair"
                else:
                    decision = "skip"

                log_event = AphasiaLogEvent(
                    decision=decision,
                    is_aphasic=aphasia_report["is_aphasic"],
                    severity=aphasia_report["severity"],
                    flags=aphasia_report["flags"],
                    detect_enabled=self.aphasia_detect_enabled,
                    repair_enabled=self.aphasia_repair_enabled,
                    severity_threshold=self.aphasia_severity_threshold,
                )
                log_aphasia_event(log_event)

        result = {
            "response": final_response,
            "phase": state["phase"],
            "accepted": True,
            "neuro_enhancement": neuro_response,
            "aphasia_flags": aphasia_report,
        }

        if governed_metadata is not None:
            result["speech_governance"] = governed_metadata

        return result
