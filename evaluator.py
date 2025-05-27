"""evaluator.py

Evaluator for the trained Transformer model. Performs beam-search decoding
on the test split and computes sacreBLEU.
"""

import os
import glob
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sacrebleu

import utils


class TestDataset(Dataset):
    """
    Simple Dataset for holding tokenized source sequences and reference texts.
    Each item is a dict with:
        - input_ids: torch.LongTensor of shape (seq_len,)
        - reference: str
    """

    def __init__(self, src_ids_list: List[List[int]], references: List[str]):
        assert len(src_ids_list) == len(references), \
            "Source/Reference length mismatch"
        self.src_ids_list = src_ids_list
        self.references = references

    def __len__(self) -> int:
        return len(self.src_ids_list)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "input_ids": torch.tensor(self.src_ids_list[idx], dtype=torch.long),
            "reference": self.references[idx]
        }


class Evaluator:
    """
    Evaluator to load a trained Transformer checkpoint, run beam search
    on the test set, and compute BLEU.
    """

    def __init__(self, model: torch.nn.Module, tokenizer, config: object):
        """
        Args:
            model: Trained TransformerModel (nn.Module).
            tokenizer: Subword tokenizer (tokenizers.Tokenizer).
            config: Config object with .get(section) -> dict.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Load latest checkpoint
        logging_cfg = config.get("logging")
        ckpt_dir = logging_cfg.get("checkpoint_dir", "checkpoints/")
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        utils.load_checkpoint(latest_ckpt, self.model)
        
        # Prepare test dataset
        data_cfg = config.get("data")
        test_src = data_cfg["test_src"]
        test_tgt = data_cfg["test_tgt"]

        # Read raw lines
        src_lines = []
        with open(test_src, "r", encoding="utf-8") as f_src:
            for line in f_src:
                line = line.strip()
                if line:
                    src_lines.append(line)
        ref_lines = []
        with open(test_tgt, "r", encoding="utf-8") as f_tgt:
            for line in f_tgt:
                line = line.strip()
                if line:
                    ref_lines.append(line)
        if len(src_lines) != len(ref_lines):
            raise ValueError("Test source/target line count mismatch")

        # Tokenize source lines
        self.src_ids_list: List[List[int]] = []
        for line in src_lines:
            enc = tokenizer.encode(line)
            self.src_ids_list.append(enc.ids)

        # Build dataset & loader
        self.test_dataset = TestDataset(self.src_ids_list, ref_lines)
        # Use batch_size=1 for per-example beam search
        self.batch_size = 1
        self.test_loader = utils.make_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Inference parameters
        infer_cfg = config.get("inference")
        self.beam_size: int = int(infer_cfg.get("beam_size", 4))
        self.length_penalty: float = float(infer_cfg.get("length_penalty", 0.6))
        self.max_offset: int = int(infer_cfg.get("max_length_offset", 50))

        # Special token IDs
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        if self.bos_id is None or self.eos_id is None:
            raise ValueError("Tokenizer missing <s> or </s> tokens")

        # Pad token index assumed 0 in model internals
        self.pad_idx = 0

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """
        Run beam search decoding on the test set and compute BLEU.

        Args:
            split: must be "test".

        Returns:
            Dict with 'BLEU' score.
        """
        if split != "test":
            raise ValueError(f"Unsupported split '{split}', only 'test' allowed")

        all_hyps: List[str] = []
        all_refs: List[str] = []

        for batch in tqdm(self.test_loader, desc="Decoding"):
            # batch["input_ids"]: (1, seq_len)
            src_ids = batch["input_ids"].to(self.device)
            reference = batch["reference"][0]  # list of one str

            # Generate best output sequence of token IDs (including specials)
            best_seq = self._generate_beam(src_ids)

            # Decode to text, skip special tokens
            hyp_text = self.tokenizer.decode(best_seq, skip_special_tokens=True)
            all_hyps.append(hyp_text)
            all_refs.append(reference)

        # Compute corpus BLEU
        bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs], force=True)
        return {"BLEU": bleu.score}

    def _generate_beam(self, src_ids: torch.Tensor) -> List[int]:
        """
        Beam-search decoding for a single source sequence.

        Args:
            src_ids: Tensor of shape (1, src_len).

        Returns:
            List of output token IDs (includes <s> and </s>).
        """
        # Compute source mask (pad=True where id!=pad_idx)
        src_mask = src_ids.ne(self.pad_idx).unsqueeze(1).unsqueeze(2)
        # Determine actual source length (non-pad tokens)
        src_len = int(src_ids.ne(self.pad_idx).sum().item())
        max_len = src_len + self.max_offset

        # Initial beam with only BOS
        beams = [([self.bos_id], 0.0)]
        completed: List[tuple] = []

        for _ in range(max_len):
            all_candidates: List[tuple] = []

            # Expand each beam
            for seq, score in beams:
                # If already ended, move to completed
                if seq[-1] == self.eos_id:
                    norm_score = score / (len(seq) ** self.length_penalty)
                    completed.append((seq, norm_score))
                    continue

                # Prepare decoder input and mask
                dec_input = torch.tensor([seq], dtype=torch.long, device=self.device)
                seq_len = dec_input.size(1)
                # padding mask for decoder (all non-pad)
                dec_pad_mask = dec_input.ne(self.pad_idx).unsqueeze(1).unsqueeze(2)
                # causal mask
                subsequent = torch.tril(
                    torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool)
                )
                dec_mask = dec_pad_mask & subsequent.unsqueeze(0)

                # Forward pass
                with torch.no_grad():
                    logits = self.model(src_ids, dec_input)  # (1, seq_len, vocab_size)
                    last_logits = logits[0, -1, :]           # (vocab_size,)
                    log_probs = F.log_softmax(last_logits, dim=-1)

                # Get top-k token candidates
                topk_logps, topk_ids = torch.topk(log_probs, self.beam_size)
                for logp, tok in zip(topk_logps.tolist(), topk_ids.tolist()):
                    new_seq = seq + [int(tok)]
                    new_score = score + float(logp)
                    all_candidates.append((new_seq, new_score))

            # If no expansions (all beams ended), stop
            if not all_candidates:
                break

            # Select top beam_size candidates
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[: self.beam_size]

            # If enough completed, we can stop early
            if len(completed) >= self.beam_size:
                break

        # If no completed hypotheses, treat remaining beams as completed
        if not completed:
            for seq, score in beams:
                norm_score = score / (len(seq) ** self.length_penalty)
                completed.append((seq, norm_score))

        # Pick hypothesis with highest normalized score
        best_seq, _ = max(completed, key=lambda x: x[1])
        return best_seq
