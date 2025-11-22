import random
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler

from ..core.cognitive_controller import CognitiveController
from ..core.llm_wrapper import LLMWrapper

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


class LanguageDataset(Dataset):
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


class CurriculumSampler(Sampler):
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


class AphasiaBrocaDetector:
    def __init__(
        self,
        min_sentence_len=6.0,
        min_function_word_ratio=0.15,
        max_fragment_ratio=0.5,
    ):
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

    def analyze(self, text):
        cleaned = text.strip()
        if not cleaned:
            return {
                "is_aphasic": True,
                "severity": 1.0,
                "avg_sentence_len": 0.0,
                "function_word_ratio": 0.0,
                "fragment_ratio": 1.0,
                "flags": ["empty_text"],
            }
        sentences = re.split(r"[.!?]+", cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return {
                "is_aphasic": True,
                "severity": 1.0,
                "avg_sentence_len": 0.0,
                "function_word_ratio": 0.0,
                "fragment_ratio": 1.0,
                "flags": ["no_sentence_boundaries"],
            }
        total_tokens = 0
        function_tokens = 0
        fragment_sentences = 0
        for sent in sentences:
            tokens = [t.lower() for t in sent.split() if t.strip()]
            length = len(tokens)
            total_tokens += length
            if length < 4:
                fragment_sentences += 1
            function_tokens += sum(1 for t in tokens if t in self.function_words)
        if total_tokens == 0:
            return {
                "is_aphasic": True,
                "severity": 1.0,
                "avg_sentence_len": 0.0,
                "function_word_ratio": 0.0,
                "fragment_ratio": 1.0,
                "flags": ["no_tokens"],
            }
        avg_sentence_len = total_tokens / len(sentences)
        function_word_ratio = function_tokens / total_tokens
        fragment_ratio = fragment_sentences / len(sentences)
        flags = []
        if avg_sentence_len < self.min_sentence_len:
            flags.append("short_sentences")
        if function_word_ratio < self.min_function_word_ratio:
            flags.append("low_function_words")
        if fragment_ratio > self.max_fragment_ratio:
            flags.append("high_fragment_ratio")
        is_aphasic = bool(flags)
        severity = 0.0
        if flags:
            severity = min(
                1.0,
                (
                    max(0.0, self.min_sentence_len - avg_sentence_len) / self.min_sentence_len
                    + max(0.0, self.min_function_word_ratio - function_word_ratio) / max(
                        self.min_function_word_ratio, 1e-6
                    )
                    + max(0.0, fragment_ratio - self.max_fragment_ratio) / max(
                        self.max_fragment_ratio, 1e-6
                    )
                )
                / 3.0,
            )
        return {
            "is_aphasic": is_aphasic,
            "severity": float(severity),
            "avg_sentence_len": float(avg_sentence_len),
            "function_word_ratio": float(function_word_ratio),
            "fragment_ratio": float(fragment_ratio),
            "flags": flags,
        }


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
    ):
        super().__init__(
            llm_generate_fn=llm_generate_fn,
            embedding_fn=embedding_fn,
            dim=dim,
            capacity=capacity,
            wake_duration=wake_duration,
            sleep_duration=sleep_duration,
            initial_moral_threshold=initial_moral_threshold,
        )
        self.dataset = LanguageDataset(all_sentences)
        vocab_size = len(self.dataset.vocab)

        self.actor = InnateGrammarModule(vocab_size)
        self.critic = InnateGrammarModule(vocab_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(device)
        self.critic.to(device)

        self.trainer = CriticalPeriodTrainer(self.actor, self.critic, self.dataset, epochs=3)
        self.trainer.train()

        self.processor1 = ModularLanguageProcessor(self.actor, self.critic, self.dataset)
        self.processor2 = ModularLanguageProcessor(self.critic, self.actor, self.dataset)
        self.integrator = SocialIntegrator(self.processor1, self.processor2)

        self.controller = CognitiveController(dim)
        self.aphasia_detector = AphasiaBrocaDetector()

    def generate(self, prompt: str, moral_value: float = 0.5, max_tokens: int = 50) -> dict:
        neuro_response = self.integrator.interact(prompt, prompt)
        embedding = self.embedding_fn(neuro_response)
        state = self.controller.process_event(embedding, moral_value)

        if not state["accepted"]:
            return {
                "response": "Rejected by moral filter.",
                "phase": state["phase"],
                "accepted": False,
                "neuro_enhancement": neuro_response,
                "aphasia_flags": None,
            }

        enhanced_prompt = f"{prompt}\n\n[NeuroLang enhancement]: {neuro_response}"
        base_response = self.llm_generate_fn(enhanced_prompt, max_tokens)

        aphasia_report = self.aphasia_detector.analyze(base_response)

        if aphasia_report["is_aphasic"]:
            repair_prompt = (
                f"{prompt}\n\nThe following draft answer shows Broca-like aphasia "
                f"(telegraphic style, broken syntax). Rewrite it in coherent, full sentences, "
                f"preserving all technical details and reasoning steps.\n\nDraft answer:\n{base_response}"
            )
            final_response = self.llm_generate_fn(repair_prompt, max_tokens)
        else:
            final_response = base_response

        return {
            "response": final_response,
            "phase": state["phase"],
            "accepted": True,
            "neuro_enhancement": neuro_response,
            "aphasia_flags": aphasia_report,
        }
