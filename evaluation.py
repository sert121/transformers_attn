## evaluation.py

import os
from typing import List, Dict

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm
import sacrebleu

from utils import Config


class Evaluator:
    """
    Evaluator for the Transformer model.
    Performs beam-search decoding on the test set and computes corpus BLEU.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        config: Config
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            model: Trained TransformerModel (in eval mode).
            test_loader: DataLoader yielding batches of {'src': Tensor, 'tgt': Tensor}.
            config: Configuration object.
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config

        # Decoding hyperparameters
        self.beam_size: int = int(self.config.get("decoding.beam_size", 4))
        self.length_penalty: float = float(self.config.get("decoding.length_penalty", 0.6))
        self.max_offset: int = int(self.config.get("decoding.max_output_length_offset", 50))

        # Load target tokenizer to decode token IDs to text
        tok_dir: str = self.config.get("data.tokenizer_dir", "tokenizers")
        vocab_type: str = self.config.get("data.vocab_type", "bpe")
        tgt_tok_file = os.path.join(tok_dir, f"tgt_{vocab_type}_tokenizer.json")
        if not os.path.isfile(tgt_tok_file):
            raise FileNotFoundError(f"Target tokenizer not found at {tgt_tok_file}")
        self.tokenizer: Tokenizer = Tokenizer.from_file(tgt_tok_file)

        # Special token IDs
        self.bos_id: int = self.tokenizer.token_to_id("<s>")
        self.eos_id: int = self.tokenizer.token_to_id("</s>")

        # Read reference lines for test set
        test_tgt_path: str = self.config.get("data.test_tgt")
        if not os.path.isfile(test_tgt_path):
            raise FileNotFoundError(f"Test reference file not found: {test_tgt_path}")
        with open(test_tgt_path, "r", encoding="utf-8") as f:
            self.references: List[str] = [line.strip() for line in f]

        # Pointer into reference list
        self._ref_index: int = 0

        # Device
        self.device = next(self.model.parameters()).device
        self.model.eval()

    def evaluate(self) -> Dict[str, float]:
        """
        Decode the entire test set and compute corpus BLEU.

        Returns:
            A dict with key 'bleu' and the BLEU score.
        """
        all_hyps: List[str] = []
        all_refs: List[str] = []

        # Disable gradient computations
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", unit="batch"):
                src_batch: torch.Tensor = batch["src"].to(self.device)
                batch_size, src_len = src_batch.size()
                max_len = src_len + self.max_offset

                for i in range(batch_size):
                    single_src = src_batch[i : i + 1]  # (1, src_len)
                    # Beam-search decode this single example
                    pred_tokens = self._beam_search(single_src, max_len)

                    # Strip BOS
                    if pred_tokens and pred_tokens[0] == self.bos_id:
                        pred_tokens = pred_tokens[1:]
                    # Truncate at EOS
                    if self.eos_id in pred_tokens:
                        idx = pred_tokens.index(self.eos_id)
                        pred_tokens = pred_tokens[:idx]

                    # Decode token IDs to string
                    hyp = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)

                    # Reference string
                    if self._ref_index >= len(self.references):
                        # Safety check
                        raise IndexError("Reference index out of range during evaluation.")
                    ref = self.references[self._ref_index]
                    self._ref_index += 1

                    all_hyps.append(hyp)
                    all_refs.append(ref)

        # Compute corpus BLEU
        # sacrebleu expects list of hyps and list of list of refs
        bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score
        return {"bleu": bleu}

    def _beam_search(
        self,
        src: torch.Tensor,
        max_len: int
    ) -> List[int]:
        """
        Perform beam-search decoding for a single source example.

        Args:
            src: Tensor of shape (1, src_len) of source token IDs.
            max_len: Maximum output length.

        Returns:
            List of token IDs for the best hypothesis (including BOS/EOS).
        """
        # Beam entries: dict with 'tokens' and 'logprob'
        beams = [{"tokens": [self.bos_id], "logprob": 0.0}]
        completed = []

        for _ in range(max_len):
            new_beams = []
            # Expand each beam
            for beam in beams:
                last_token = beam["tokens"][-1]
                # If already ended, carry forward
                if last_token == self.eos_id:
                    completed.append(beam)
                    continue

                # Prepare decoder input
                dec_input = torch.tensor(
                    beam["tokens"], dtype=torch.long, device=self.device
                ).unsqueeze(0)  # (1, cur_len)

                # Forward pass (model builds its own masks)
                logits = self.model(src, dec_input)  # (1, cur_len, vocab_size)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)  # (vocab_size,)

                # Top-k tokens
                topk_logp, topk_ids = torch.topk(log_probs, self.beam_size)
                for logp, idx in zip(topk_logp.tolist(), topk_ids.tolist()):
                    new_beams.append({
                        "tokens": beam["tokens"] + [int(idx)],
                        "logprob": beam["logprob"] + float(logp)
                    })

            # If no expansions (all beams completed), stop
            if not new_beams:
                break

            # Prune to beam_size by length-normalized score
            def score_norm(entry: Dict[str, float]) -> float:
                length = len(entry["tokens"])
                # avoid division by zero
                return entry["logprob"] / (length ** self.length_penalty)  

            # Select top beams
            beams = sorted(new_beams, key=score_norm, reverse=True)[: self.beam_size]

            # Early stopping if all current beams have ended
            if all(b["tokens"][-1] == self.eos_id for b in beams):
                completed.extend(beams)
                break

        # If no completed hypotheses, use current beams
        if not completed:
            completed = beams

        # Return the best completed hypothesis
        best = max(completed, key=lambda b: b["logprob"] / (len(b["tokens"]) ** self.length_penalty))
        return best["tokens"]
