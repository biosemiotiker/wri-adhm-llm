"""
wri_analysis5.py – Temperatur-Studie für WRI-Auswertung mit HuggingFace transformers Backend
==========================================================================================

VERWENDUNG:
    python wri_rating_qwen.py --help
    python wri_rating_qwen.py --input-folder data/transcripts/ --output-base results/
    python wri_rating_qwen.py --input-folder data/ --temperatures 0.0 0.4 0.8

WAS DIESES SKRIPT TUT:
    Führt eine systematische Temperaturstudie durch: n Patienten werden bei mehreren Temperaturen ausgewertet.
    Kombiniert Chris Landaus verbesserten 4-Phasen-Prompt mit Oliver Grimms TransformersBackend.
    Erzeugt pro Temperatur einen Ordner mit Individualauswertungen und einer Zusammenfassungstabelle.

AUSGABE-STRUKTUR:
    {output_base}/
        temp_0.0/
            {patient_id}.txt
            wri_results_table.csv
        temp_0.4/
            ...
        temp_1.6/
            ...

ABHÄNGIGKEITEN:
    pip install torch transformers pandas tqdm
    HuggingFace Modell muss heruntergeladen sein (oder wird automatisch geladen)
"""

import os
import re
import glob
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from types import SimpleNamespace

try:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        BNB_AVAILABLE = True
    except Exception:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        BNB_AVAILABLE = False
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    raise ImportError("transformers ist nicht installiert. Führen Sie 'pip install transformers' aus.")

# ===================================================================
# MODUL-LEVEL KONSTANTEN
# ===================================================================

DEFAULT_MODEL = "Qwen/Qwen3-7B-Instruct"  # Beispiel; override mit --model
DEFAULT_TEMPERATURES = [0.0]
DEFAULT_MAX_NEW_TOKENS = 8192

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===================================================================
# WRI_PROMPT (wird per --prompt-file geladen)
# ===================================================================

WRI_PROMPT = ""  # wird zur Laufzeit via --prompt-file geladen (Prompt aus Urheberrechtsgründen nicht im Repo)


def load_prompt(path):
    """Liest den WRI-Prompt aus einer lokalen Datei."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"Prompt-Datei nicht gefunden: {p}\n"
            "Der WRI-Prompt ist aus Urheberrechtsgründen nicht im Repository enthalten. "
            "Siehe prompts/EXAMPLE_PROMPT.md für Struktur und Ausgabeformat."
        )
    return p.read_text(encoding="utf-8")

# ===================================================================
# PARSER DEFINITIONEN (aus wri_analysis2.1.py, Zeilen 326–367)
# ===================================================================

ITEM_KEYS: List[str] = [
    "1.1","1.2","1.3","1.4","1.5",
    "2.1","2.2","2.3",
    "3.1","3.2","3.3",
    "4.1","4.2","4.3","4.4",
    "5.1","5.2","5.3",
    "6.1","6.2","6.3","6.4","6.5",
    "7.1","7.2","7.3","7.4","7.5",
]

K_SUM_KEYS = [f"K{i}_SUM" for i in range(1, 8)]
K_GB_KEYS  = [f"K{i}_GB"  for i in range(1, 8)]

K_USUM_KEYS = [f"K{i}_USUM" for i in range(1, 8)]
K_UGB_KEYS  = [f"K{i}_UGB"  for i in range(1, 8)]

# Nur Keys, die das LLM liefert (Phase 1 + 2 + 3-Unsicherheit)
# SUM_ERF, GB_ERF, ERF, DIAG werden extern berechnet
ALL_KEYS_IN_ORDER: List[str] = (
    ITEM_KEYS +
    K_SUM_KEYS + K_GB_KEYS +
    K_USUM_KEYS + K_UGB_KEYS +
    ["DIAG_U"]
)

# Anzahl Items pro Kriterium (fuer deterministische SUM_ERF-Berechnung)
CRITERION_N_ITEMS = {1: 5, 2: 3, 3: 3, 4: 4, 5: 3, 6: 5, 7: 5}

# Items pro Kriterium (fuer deterministische SUM-Berechnung)
CRITERION_ITEM_KEYS = {
    1: ["1.1", "1.2", "1.3", "1.4", "1.5"],
    2: ["2.1", "2.2", "2.3"],
    3: ["3.1", "3.2", "3.3"],
    4: ["4.1", "4.2", "4.3", "4.4"],
    5: ["5.1", "5.2", "5.3"],
    6: ["6.1", "6.2", "6.3", "6.4", "6.5"],
    7: ["7.1", "7.2", "7.3", "7.4", "7.5"],
}

def parse_wri_output_to_dict(text: str) -> Dict[str, Optional[int]]:
    """Parse WRI-Output mit [KEY]=VALUE Format.

    Args:
        text: Ausgabetext vom Modell

    Returns:
        Dict mit allen Keys als Schlüssel und int-Werten (oder None bei Parse-Fehler)
    """
    values: Dict[str, Optional[int]] = {}
    for key in ALL_KEYS_IN_ORDER:
        # Item-Keys (z.B. "1.1"): öffnende Klammer optional, da Modell manchmal "2.1)=1" ausgibt
        # GB/SUM-Keys (z.B. "K1_GB"): Klammer erwartet
        # Beide: schließende ) statt ] tolerated; case-insensitive; Wert kann in {} stehen
        if key[0].isdigit():
            # negativer lookbehind verhindert Match von "21.1)" als "1.1"
            pattern = r"(?<!\d)\[?" + re.escape(key) + r"[\])]?\s*=\s*\{?(-?\d+)\}?"
        else:
            pattern = r"\[" + re.escape(key) + r"[\])]?\s*=\s*\{?(-?\d+)\}?"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                values[key] = int(match.group(1))
            except ValueError:
                values[key] = None
        else:
            values[key] = None
    return values

# ===================================================================
# HILFSFUNKTION: ROBUSTES DATEILESEN
# ===================================================================

def read_text_file_robust(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Liest Textdatei mit mehreren Encodings.

    Args:
        path: Pfad zur Datei

    Returns:
        Tuple (Inhalt oder None, verwendetes Encoding oder None)
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read(), enc
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Fehler beim Lesen von {path} mit {enc}: {e}")
            return None, None
    # Fallback mit Ersatzzeichen
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            logger.warning(f"Warnung: {path} mit Ersatzzeichen (utf-8, errors='replace') gelesen.")
            return f.read(), "utf-8-replace"
    except Exception as e:
        logger.error(f"Fehler beim Fallback-Lesen von {path}: {e}")
        return None, None

# ===================================================================
# TRANSKRIPT-SUCHE (aus wri_analysis2.1.py, Zeilen 299–321)
# ===================================================================

def find_transcripts_production(input_folder: str) -> Dict[str, str]:
    """Durchsucht Produktions-Ordner nach Transkripten.

    Args:
        input_folder: Basis-Ordner mit adhs_* Unterordnern

    Returns:
        Dict {patient_id: filepath}
    """
    transcript_paths: Dict[str, str] = {}
    for entry in sorted(os.listdir(input_folder)):
        folder_path = os.path.join(input_folder, entry)
        if os.path.isdir(folder_path) and entry.lower().startswith("adhs"):
            id_match = re.match(r"^adhs_(.+?)(?:-corrected)?$", entry, re.IGNORECASE)
            patient_id = id_match.group(1) if id_match else entry
            matches = glob.glob(os.path.join(folder_path, "*corrected.txt"))
            if matches:
                transcript_paths[patient_id] = matches[0]
            else:
                logger.warning(f"Kein 'corrected.txt' in: {folder_path}")
    return transcript_paths


def find_transcripts_test(test_folder: str) -> Dict[str, str]:
    """Durchsucht Test-Ordner nach Transkripten.

    Args:
        test_folder: Test-Ordner mit *-corrected.txt Dateien

    Returns:
        Dict {patient_id: filepath}
    """
    transcript_paths: Dict[str, str] = {}
    matches = glob.glob(os.path.join(test_folder, "*-corrected.txt"))
    for filepath in sorted(matches):
        filename = os.path.basename(filepath)
        patient_id = re.sub(r"-corrected\.txt$", "", filename, flags=re.IGNORECASE)
        transcript_paths[patient_id] = filepath
    return transcript_paths

# ===================================================================
# TRANSFORMERS BACKEND (aus wri_analysis4.py, Zeilen 436–596)
# ===================================================================

class TransformersBackend:
    """Wrapper für HuggingFace transformers-basierte Inferenz mit Token-Logprobs."""

    def __init__(self, model_id: str, device_map="auto", max_new_tokens=4096):
        """
        Args:
            model_id: HuggingFace Hub-ID oder lokaler Pfad (z.B. 'openai/gpt-oss-20b')
            device_map: 'auto' oder spezifischer Device-Name
            max_new_tokens: Maximale generierte Tokens
        """
        logger.info(f"Lade Modell: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Bevorzugt 8-Bit mit CPU-Offload, falls verfügbar
        model_kwargs = dict(device_map=device_map)
        if 'torch' in globals():
            model_kwargs['torch_dtype'] = torch.float16
        if 'BNB_AVAILABLE' in globals() and BNB_AVAILABLE:
            try:
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model_kwargs['quantization_config'] = quant_cfg
                logger.info("Verwende 8-bit-Quantisierung")
            except Exception:
                pass
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
        except ValueError as e:
            if "quantization_config" in str(e) or "Mxfp4Config" in str(e) or "quantized" in str(e).lower():
                # Modell ist bereits quantisiert (z.B. MXFP4) → BNB-Config entfernen und nochmal laden
                logger.info("Modell ist bereits quantisiert, entferne Quantisierung-Config")
                model_kwargs.pop("quantization_config", None)
                model_kwargs.pop("torch_dtype", None)  # MXFP4 bestimmt dtype selbst
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs
                )
            else:
                raise
        # Reasoning-Modell braucht mehr Tokens: Reasoning-Chain + 28 Items + Summen + GB + Diagnose
        self.max_new_tokens = min(max_new_tokens, 8192)
        logger.info(f"Modell erfolgreich geladen. max_new_tokens: {self.max_new_tokens}")

    def generate_with_logprobs(self, messages: list, temperature: float = 0.0, top_p: float = 0.9) -> tuple:
        """Generiert Text mit Token-Level Logprobs ohne chat_template.

        Flattened-Chat: Verkettet alle message["content"] mit Doppel-NEWLINE zu einem Plain-Prompt.
        Bei temperature > 0.0 wird Sampling (top-p) aktiviert, sonst deterministisch.
        Bewahrt den Prompt-Kopf vor Kürzung; schneidet nur das Transkript-Tail.

        Args:
            messages: Liste von dict mit 'role' und 'content'
            temperature: Sampling-Temperatur (0.0 = deterministisch)
            top_p: Top-p Sampling Parameter

        Returns:
            Tuple (generierter Text, pseudo-logprobs Liste)
        """
        # Plain-Prompt aus Nachrichten zusammenbauen
        prompt = "\n\n".join(m.get("content", "") for m in messages)

        # Heuristik: Prompt in Kopf (Anleitung) und Transkript trennen
        # PASSUNG zu Chris' Prompt: Marker ist "TRANSKRIPT:"
        marker = "TRANSKRIPT:"
        if marker in prompt:
            head, tail = prompt.split(marker, 1)
            head = head + marker  # Marker am Ende des Kopfes behalten
            # Separat tokenisieren, um Kopf garantiert zu behalten
            head_enc = self.tokenizer(head, return_tensors="pt")
            tail_enc = self.tokenizer(tail, return_tensors="pt")
            head_ids = head_enc["input_ids"].to(self.model.device)
            tail_ids = tail_enc["input_ids"].to(self.model.device)

            max_ctx = getattr(self.model.config, "max_position_embeddings", 2048)
            reserve = max(64, min(self.max_new_tokens, 256))
            allowed_input = max(64, max_ctx - reserve)
            # Platz für Transkript bestimmen
            keep_tail = max(0, allowed_input - head_ids.shape[1])
            if keep_tail > 0:
                input_ids = torch.cat([head_ids, tail_ids[:, -keep_tail:]], dim=1)
            else:
                # Fallback: Aufteilung Kopf/Tail 50/50
                split_point = allowed_input // 2
                head_part = head_ids[:, -split_point:]
                tail_part = tail_ids[:, -(allowed_input - split_point):]
                input_ids = torch.cat([head_part, tail_part], dim=1)
            # Attention mask bauen
            attention_mask = torch.ones_like(input_ids)
            gen_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            input_len = input_ids.shape[1]
        else:
            enc = self.tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            input_ids = enc["input_ids"]

            # Kürze Eingabe auf Modellkontext, damit Generation stabil bleibt
            max_ctx = getattr(self.model.config, "max_position_embeddings", 2048)
            reserve = max(64, min(self.max_new_tokens, 256))
            allowed_input = max(64, max_ctx - reserve)
            if input_ids.shape[1] > allowed_input:
                input_ids = input_ids[:, -allowed_input:]
                if "attention_mask" in enc:
                    enc["attention_mask"] = enc["attention_mask"][:, -allowed_input:]
            input_len = input_ids.shape[1]
            gen_inputs = {"input_ids": input_ids}
            if "attention_mask" in enc:
                gen_inputs["attention_mask"] = enc["attention_mask"]

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.15,  # erhöht um Reasoning-Loops zu dämpfen
            no_repeat_ngram_size=5,   # bricht exakte 5-Token-Loops sofort ab
        )
        if temperature and float(temperature) > 0.0:
            gen_kwargs.update(dict(
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
            ))
        else:
            gen_kwargs.update(dict(do_sample=False))

        with torch.no_grad():
            outputs = self.model.generate(**gen_inputs, **gen_kwargs)

        text = self.tokenizer.decode(
            outputs.sequences[0, input_len:],
            skip_special_tokens=True
        )

        pseudo_logprobs = _hf_scores_to_logprob_objects(
            outputs.scores,
            outputs.sequences[0],
            input_len,
            self.tokenizer
        )

        return text, pseudo_logprobs


def _hf_scores_to_logprob_objects(scores, sequences, input_len, tokenizer, top_k=10):
    """Adapter: Wandelt HuggingFace outputs.scores in Pseudo-TokenLogprob-Objekte um."""
    result = []

    for t, logits in enumerate(scores):
        log_probs = F.log_softmax(logits[0], dim=-1)
        chosen_id = sequences[input_len + t].item()
        chosen_logprob = log_probs[chosen_id].item()
        chosen_token = tokenizer.convert_ids_to_tokens(chosen_id)

        top_values, top_ids = torch.topk(log_probs, k=min(top_k, log_probs.shape[0]))
        top_probs = F.softmax(top_values, dim=-1).tolist()
        top_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_ids]
        top_logprobs_list = top_values.tolist()

        # Pseudo-Logprob-Objekt (kompatibel mit LM Studio API)
        result.append({
            "token": chosen_token,
            "logprob": chosen_logprob,
            "top_logprobs": [
                {"token": tok, "logprob": lp}
                for tok, lp in zip(top_tokens, top_logprobs_list)
            ],
            "text_offset": t  # Platzhalter
        })

    return result


# ===================================================================
# LM STUDIO BACKEND
# ===================================================================

class LMStudioBackend:
    """Backend für GGUF-Modelle via LM Studio API (lmstudio Python-Paket).

    LM Studio muss laufen (Port 1234). Das Modell wird beim ersten Aufruf
    automatisch in LM Studio geladen.
    """

    def __init__(self, model_name: str, context_length: int = 40000,
                 max_tokens: int = 8192):
        import lmstudio as lms
        self.model_name = model_name
        self.context_length = context_length
        self.max_tokens = max_tokens
        self._lms = lms
        # Erhöhe Timeout für große Modelle (Default: 60s — zu kurz für 235B)
        lms.set_sync_api_timeout(600.0)
        logger.info(f"LMStudioBackend: model={model_name}, "
                    f"ctx={context_length}, max_tokens={max_tokens}, timeout=600s")

    def generate_with_logprobs(self, messages: list, temperature: float = 0.0,
                               top_p: float = 0.9) -> tuple:
        """Ruft LM Studio API auf. Gibt (text, None) zurück (keine Logprobs)."""
        import re as _re
        system_text = None
        user_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                user_text = content

        with self._lms.Client() as client:
            model = client.llm.model(
                self.model_name,
                config={"contextLength": self.context_length}
            )
            chat = self._lms.Chat(system_text) if system_text else self._lms.Chat()
            chat.add_user_message(user_text)

            config = {
                "temperature": float(temperature),
                "topP": float(top_p) if float(temperature) > 0.0 else 1.0,
                "topK": 0,
                "maxTokens": self.max_tokens,
            }
            response = model.respond(chat, config=config)
            raw = response.content if hasattr(response, "content") else str(response)

        # LM Studio Artefakte + Qwen3 <think>-Blöcke entfernen
        raw = _re.sub(r"<\|channel\|>.*?<\|end\|>", "", raw, flags=_re.DOTALL)
        raw = _re.sub(r"<\|[^|]+\|>", "", raw)
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL)
        return raw.strip(), None


# ===================================================================
# OPENAI BACKEND (für vLLM, Ollama, etc.)
# ===================================================================

class OpenAIBackend:
    """Backend für OpenAI-kompatible APIs (vLLM, Ollama, etc.)."""

    def __init__(self, base_url: str, model_name: str, max_tokens: int = 8192):
        """
        Args:
            base_url: Base-URL der OpenAI-kompatiblen API (z.B. http://localhost:8001/v1)
            model_name: Name des zu verwendenden Modells
            max_tokens: Maximale generierte Tokens
        """
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="vllm")
        self.model_name = model_name
        self.max_tokens = max_tokens
        logger.info(f"OpenAIBackend: base_url={base_url}, model={model_name}, max_tokens={max_tokens}")

    def generate_with_logprobs(self, messages: list, temperature: float = 0.0, top_p: float = 0.9) -> tuple:
        """Ruft OpenAI-kompatible API auf."""
        import re as _re
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p if temperature > 0.0 else 1.0,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content or ""
        # Qwen3 <think>-Blöcke entfernen
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        return text, None


# ===================================================================
# OLLAMA BACKEND
# ===================================================================

class OllamaBackend:
    """Backend für Ollama REST API."""

    def __init__(self, ollama_host: str, model_name: str, max_tokens: int = 8192):
        """
        Args:
            ollama_host: Ollama Host URL (z.B. http://127.0.0.1:11434)
            model_name: Name des zu verwendenden Modells
            max_tokens: Maximale generierte Tokens
        """
        import requests
        self.ollama_host = ollama_host.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.requests = requests
        logger.info(f"OllamaBackend: host={ollama_host}, model={model_name}, max_tokens={max_tokens}")

    def generate_with_logprobs(self, messages: list, temperature: float = 0.0, top_p: float = 0.9) -> tuple:
        """Ruft Ollama REST API auf."""
        import re as _re

        # Convert messages to format Ollama expects
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{content}\n\n"

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p if temperature > 0.0 else 1.0,
            "stream": False,
            "num_predict": self.max_tokens,
        }

        try:
            response = self.requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "").strip()
            # Qwen3 <think>-Blöcke entfernen
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
            return text, None
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


# ===================================================================
# CSV-BUILDER (aus wri_analysis2.1.py, Zeilen 369–440)
# ===================================================================

def build_results_csv(output_folder: str) -> None:
    """Erstellt wri_results_table.csv aus allen .txt-Dateien im Ordner.

    Args:
        output_folder: Ordner mit WRI-Output-Dateien
    """
    logger.info(f"\n--- Erstelle wri_results_table.csv ---")

    wri_files = sorted(
        f for f in os.listdir(output_folder)
        if f.endswith(".txt") and not f.endswith("_confidence.txt")
    )
    if not wri_files:
        logger.warning("Keine WRI-Output-Dateien gefunden.")
        return

    all_rows: List[Dict[str, Optional[int]]] = []

    for filename in wri_files:
        patient_id = os.path.splitext(filename)[0]
        wri_path = os.path.join(output_folder, filename)

        try:
            with open(wri_path, "r", encoding="utf-8") as f:
                wri_content = f.read()
        except UnicodeDecodeError:
            wri_content, enc_used = read_text_file_robust(wri_path)
            if wri_content is None:
                logger.error(f"Fehler beim Lesen von {wri_path}, Datei wird übersprungen.")
                continue
        except Exception as e:
            logger.error(f"Fehler beim Lesen von {wri_path}: {e}")
            continue

        parsed = parse_wri_output_to_dict(wri_content)
        row: Dict[str, Optional[int]] = {"patient_id": patient_id}

        # Phase 1: Items (vom LLM)
        for key in ITEM_KEYS:
            row[f"item_{key.replace('.', '_')}"] = parsed[key]

        # Phase 2: GB vom LLM, SUM deterministisch berechnet
        for k in range(1, 8):
            # SUM: extern aus Items berechnen (statt LLM-Wert zu vertrauen)
            item_vals = [parsed[ik] for ik in CRITERION_ITEM_KEYS[k]]
            valid_vals = [v for v in item_vals if v is not None and v >= 0]
            computed_sum = sum(valid_vals) if valid_vals else None
            row[f"k{k}_sum"] = computed_sum

            # LLM-SUM als Vergleich speichern (QC)
            llm_sum = parsed.get(f"K{k}_SUM")
            row[f"k{k}_sum_llm"] = llm_sum
            if computed_sum is not None and llm_sum is not None and computed_sum != llm_sum:
                logger.debug(f"  {patient_id} K{k}: SUM computed={computed_sum} vs LLM={llm_sum}")

            # GB vom LLM uebernehmen (klinisches Urteil, nicht berechenbar)
            row[f"k{k}_gb"] = parsed[f"K{k}_GB"]

        # Deterministische Berechnung: SUM_ERF, GB_ERF, ERF, DIAG
        for k in range(1, 8):
            s = row[f"k{k}_sum"]
            gb = row[f"k{k}_gb"]
            n_items = CRITERION_N_ITEMS[k]
            row[f"k{k}_sum_erf"] = int(s >= n_items) if s is not None else None
            row[f"k{k}_gb_erf"] = int(gb >= 2) if gb is not None else None
            se = row[f"k{k}_sum_erf"]
            ge = row[f"k{k}_gb_erf"]
            row[f"k{k}_erf"] = int(se == 1 and ge == 1) if (se is not None and ge is not None) else None

        # DIAG: K1+K2 erfuellt UND mindestens 2 von K3-K7 erfuellt
        k1_erf = row.get("k1_erf")
        k2_erf = row.get("k2_erf")
        extra_erf = [row.get(f"k{k}_erf") for k in range(3, 8)]
        extra_count = sum(1 for v in extra_erf if v == 1)
        if k1_erf is not None and k2_erf is not None and all(v is not None for v in extra_erf):
            row["diag"] = int(k1_erf == 1 and k2_erf == 1 and extra_count >= 2)
        else:
            row["diag"] = None

        # Phase 3 (Unsicherheit, vom LLM)
        for k in range(1, 8):
            row[f"k{k}_usum"] = parsed[f"K{k}_USUM"]
            row[f"k{k}_ugb"]  = parsed[f"K{k}_UGB"]
        row["diag_u"] = parsed["DIAG_U"]

        all_rows.append(row)

    if not all_rows:
        logger.warning("Keine Daten für CSV gefunden.")
        return

    df = pd.DataFrame(all_rows).set_index("patient_id")

    csv_path = os.path.join(output_folder, "wri_results_table.csv")
    df.to_csv(csv_path)
    logger.info(f"CSV gespeichert: {csv_path}")
    logger.info(f"Dimensionen: {df.shape[0]} Probanden x {df.shape[1]} Spalten")

    # Qualitätskontrolle
    nan_count = df.isna().sum().sum()
    minus_one_count = (df == -1).sum().sum()
    logger.info(f"Qualitätskontrolle: {nan_count} nicht geparste Werte (NaN), "
          f"{minus_one_count} Einträge mit -1 (nicht beurteilbar).")

    # SUM-Vergleich: LLM vs. berechnet
    sum_mismatch = 0
    for k in range(1, 8):
        computed = df.get(f"k{k}_sum")
        llm = df.get(f"k{k}_sum_llm")
        if computed is not None and llm is not None:
            mask = computed.notna() & llm.notna() & (computed != llm)
            sum_mismatch += mask.sum()
    if sum_mismatch > 0:
        logger.warning(f"SUM-Diskrepanzen (LLM vs. berechnet): {sum_mismatch}")
    else:
        logger.info("SUM-Konsistenz: Alle LLM-Summen stimmen mit berechneten überein.")

    # Parse-Erfolgsrate: Anteil Patienten mit allen 7 GB valide
    gb_cols = [f"k{k}_gb" for k in range(1, 8)]
    gb_valid = df[gb_cols].notna().all(axis=1).sum()
    logger.info(f"Parse-Erfolgsrate: {gb_valid}/{len(df)} ({100*gb_valid/len(df):.1f}%) mit allen 7 GB valide.")

# ===================================================================
# TEMPERATUR-SWEEP (neu)
# ===================================================================

def temperature_sweep(
    backend: TransformersBackend,
    patient_transcripts: Dict[str, str],
    temperatures: List[float],
    output_base: str,
) -> None:
    """Führt Temperatur-Sweep durch: mehrere Temperaturen pro Patient.

    Args:
        backend: TransformersBackend-Instanz
        patient_transcripts: Dict {patient_id: transcript_content}
        temperatures: Liste der zu testenden Temperaturen
        output_base: Basisordner für Outputs
    """
    logger.info(f"\n=== TEMPERATUR-SWEEP ===")
    logger.info(f"Patienten: {len(patient_transcripts)}")
    logger.info(f"Temperaturen: {temperatures}")
    logger.info(f"Output-Basis: {output_base}")

    total_runs = len(patient_transcripts) * len(temperatures)
    completed_runs = 0

    for temp in temperatures:
        output_dir = os.path.join(output_base, f"temp_{temp}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"\n--- Temperatur {temp} ---")
        logger.info(f"Output-Ordner: {output_dir}")

        for patient_id, transcript in tqdm(patient_transcripts.items(),
                                          desc=f"Temp {temp}"):
            out_file = os.path.join(output_dir, f"{patient_id}.txt")

            # Resume-Logik: Überspringe fertige Patienten
            if os.path.exists(out_file):
                logger.debug(f"Überspringe {patient_id} (Datei existiert bereits)")
                completed_runs += 1
                continue

            prompt = WRI_PROMPT + f"\n\nTRANSKRIPT:\n\n{transcript}"
            if isinstance(backend, LMStudioBackend):
                # Qwen3: /no_think deaktiviert Thinking-Mode, verbessert Format-Compliance
                messages = [
                    {"role": "system",
                     "content": "/no_think\nGib ausschließlich strukturierte [KEY]=WERT Ausgaben im vorgegebenen Format aus."},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            try:
                logger.info(f"Verarbeite {patient_id} bei Temp {temp}...")
                text, logprobs = backend.generate_with_logprobs(
                    messages, temperature=temp
                )
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)
                logger.info(f"✓ {patient_id} abgeschlossen ({len(text)} Zeichen)")
                completed_runs += 1
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    logger.warning(f"⚠ OOM bei {patient_id}, temp={temp}: {e}")
                    # OOM als echter Fehler betrachten - nicht incrementieren
                else:
                    logger.error(f"✗ Runtime Error bei {patient_id}, temp={temp}: {e}")
                    raise
            except Exception as e:
                import traceback as _tb
                logger.error(f"✗ Fehler bei {patient_id}, temp={temp}: {e}\n{_tb.format_exc()}")
                # Timeout und Server-Fehler: Patient überspringen, weitermachen
                err_str = str(type(e).__name__)
                if any(x in err_str for x in ("Timeout", "ServerError", "ConnectionError")):
                    logger.warning(f"  → Überspringe {patient_id} (wiederholbarer Fehler)")
                else:
                    raise

        # CSV nach jeder Temperatur erstellen
        build_results_csv(output_dir)

        logger.info(f"✓ Temperatur {temp} vollständig: {completed_runs}/{total_runs} runs insgesamt")

    logger.info(f"\n=== TEMPERATUR-SWEEP ABGESCHLOSSEN ===")
    logger.info(f"Alle Ergebnisse in: {output_base}")

# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WRI-Temperatur-Studie mit HuggingFace transformers Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python wri_rating_qwen.py --help
  python wri_rating_qwen.py --input-folder data/ --output-base results/ --temperatures 0.0
  python wri_rating_qwen.py --input-folder data/ --temperatures 0.0 0.4 0.8
  python wri_rating_qwen.py --input-folder examples/ --model Qwen/Qwen3-7B-Instruct --temperatures 0.0
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-7B-Instruct",
        help="HuggingFace Model-ID (default: Qwen/Qwen3-7B-Instruct)"
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.0],
        help="Liste von Temperaturen (default: [0.0])"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Ordner mit Transkript-Dateien (z.B. data/ oder examples/)"
    )
    parser.add_argument(
        "--patients",
        type=str,
        help="Kommaseparierte Liste von Patienten-IDs (optional, sonst alle)"
    )
    parser.add_argument(
        "--patients-file",
        type=str,
        help="CSV-Datei mit Spalte 'patient_id' und 'selected'==True (aus select_patients.py)"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="results/",
        help="Basisordner für Outputs (default: results/)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Max generierte Tokens (default: 8192)"
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "lmstudio", "openai", "ollama"],
        default="transformers",
        help="Inferenz-Backend: 'transformers' (HuggingFace), 'lmstudio' (GGUF via LM Studio), 'openai' (vLLM), oder 'ollama' (Ollama)"
    )
    parser.add_argument(
        "--lmstudio-model",
        type=str,
        default=None,
        help="LM Studio Modellname (z.B. 'qwen3-32b'), nur für --backend lmstudio"
    )
    parser.add_argument(
        "--lmstudio-context",
        type=int,
        default=40000,
        help="LM Studio Kontextlänge in Tokens (default: 40000)"
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Lokale Datei mit dem WRI-Prompt (nicht im Repo enthalten, siehe prompts/EXAMPLE_PROMPT.md)"
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="Base-URL der OpenAI-kompatiblen API (z.B. http://localhost:8001/v1), nur für --backend openai"
    )
    parser.add_argument(
        "--openai-model-name",
        type=str,
        default=None,
        help="Modellname für OpenAI-API (z.B. 'qwen3-32b'), nur für --backend openai"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama Host URL (z.B. http://127.0.0.1:11434), nur für --backend ollama"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Ollama Modellname (z.B. 'qwen3-32b'), nur für --backend ollama"
    )

    args = parser.parse_args()

    globals()["WRI_PROMPT"] = load_prompt(args.prompt_file)
    input_folder = args.input_folder

    logger.info("=" * 80)
    logger.info("WRI-Auswertung mit HuggingFace transformers Backend")
    logger.info("=" * 80)
    logger.info(f"Modell: {args.model}")
    logger.info(f"Temperaturen: {args.temperatures}")
    logger.info(f"Input-Ordner: {input_folder}")
    logger.info(f"Output-Basis: {args.output_base}")
    logger.info(f"Max New Tokens: {args.max_new_tokens}")
    logger.info("=" * 80)

    # Transkripte suchen
    transcript_paths = find_transcripts_production(input_folder)

    if not transcript_paths:
        logger.error(f"Keine Transkripte gefunden in: {input_folder}")
        return

    # Patienten filtern, falls angegeben
    if args.patients_file:
        sel_df = pd.read_csv(args.patients_file)
        if "selected" in sel_df.columns:
            sel_df = sel_df[sel_df["selected"] == True]
        patient_list = sel_df["patient_id"].astype(str).tolist()
        transcript_paths = {
            pid: path for pid, path in transcript_paths.items() if pid in patient_list
        }
        logger.info(f"Patienten aus {args.patients_file}: {len(patient_list)} geladen, {len(transcript_paths)} gefunden")
    elif args.patients:
        patient_list = [p.strip() for p in args.patients.split(",")]
        transcript_paths = {
            pid: path
            for pid, path in transcript_paths.items()
            if pid in patient_list
        }
        missing = set(patient_list) - set(transcript_paths.keys())
        if missing:
            logger.warning(f"Patienten nicht gefunden: {missing}")

    logger.info(f"\n--- {len(transcript_paths)} Transkripte gefunden ---")
    for patient_id in sorted(transcript_paths.keys()):
        logger.info(f"  - {patient_id}")

    # Transkripte in den Speicher laden
    patient_transcripts = {}
    for patient_id, filepath in transcript_paths.items():
        transcript_content, enc_used = read_text_file_robust(filepath)
        if transcript_content is None:
            logger.error(f"Überspringe {filepath} wegen Lesefehler.")
            continue
        patient_transcripts[patient_id] = transcript_content
        logger.debug(f"{patient_id}: {len(transcript_content)} Zeichen, Encoding: {enc_used}")

    if not patient_transcripts:
        logger.error("Keine gültigen Transkripte geladen.")
        return

    # Backend initialisieren
    if args.backend == "lmstudio":
        if not args.lmstudio_model:
            parser.error("--lmstudio-model ist für --backend lmstudio erforderlich")
        backend = LMStudioBackend(
            model_name=args.lmstudio_model,
            context_length=args.lmstudio_context,
            max_tokens=args.max_new_tokens,
        )
    elif args.backend == "openai":
        if not args.openai_base_url or not args.openai_model_name:
            parser.error("--openai-base-url und --openai-model-name sind für --backend openai erforderlich")
        backend = OpenAIBackend(
            base_url=args.openai_base_url,
            model_name=args.openai_model_name,
            max_tokens=args.max_new_tokens,
        )
    elif args.backend == "ollama":
        if not args.ollama_model:
            parser.error("--ollama-model ist für --backend ollama erforderlich")
        backend = OllamaBackend(
            ollama_host=args.ollama_host,
            model_name=args.ollama_model,
            max_tokens=args.max_new_tokens,
        )
    else:
        backend = TransformersBackend(
            model_id=args.model,
            max_new_tokens=args.max_new_tokens
        )

    # Temperatur-Sweep ausführen
    temperature_sweep(
        backend=backend,
        patient_transcripts=patient_transcripts,
        temperatures=args.temperatures,
        output_base=args.output_base
    )

    logger.info("\n" + "=" * 80)
    logger.info("VOLLSTÄNDIG ABGESCHLOSSEN")
    logger.info("=" * 80)
    logger.info(f"Alle Ergebnisse in: {args.output_base}")
    logger.info("Untersuchen Sie die CSVs für die Analyse:")
    for temp in args.temperatures:
        csv_path = os.path.join(args.output_base, f"temp_{temp}", "wri_results_table.csv")
        logger.info(f"  - Temperatur {temp}: {csv_path}")


if __name__ == "__main__":
    main()