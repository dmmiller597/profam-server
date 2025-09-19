import os
import sys
import torch
import runpod
import traceback
import tempfile
import rootutils

# Add ProFam to path
sys.path.insert(0, '/workspace/profam')
rootutils.setup_root('/workspace/profam', indicator=".project-root", pythonpath=True)

from src.models.base import load_checkpoint
from src.models.inference import (
    EnsemblePromptBuilder,
    ProFamEnsembleSampler,
    PromptBuilder,
    ProFamSampler
)
from src.data.objects import ProteinDocument
from src.data.processors.preprocessing import PreprocessingConfig, ProteinDocumentPreprocessor, AlignedProteinPreprocessingConfig
from src.sequence.fasta import read_fasta
from src.models.llama import LlamaLitModule
from src.utils.utils import seed_all

MODEL_PATH = "/workspace/last.ckpt"


def write_fasta(sequences, accessions, fasta_path):
    """Write sequences to FASTA file"""
    with open(fasta_path, "w") as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


def read_fasta_from_string(fasta_content):
    """Read FASTA content from string instead of file"""
    sequences = []
    accessions = []
    current_seq = []
    current_acc = None
    
    for line in fasta_content.strip().split('\n'):
        if line.startswith('>'):
            if current_acc is not None:
                sequences.append(''.join(current_seq))
            current_acc = line[1:].strip()
            accessions.append(current_acc)
            current_seq = []
        else:
            current_seq.append(line.strip())
    
    if current_acc is not None:
        sequences.append(''.join(current_seq))
    
    return accessions, sequences


def build_pool_from_sequences(sequences, accessions, identifier="input", is_msa=True):
    """Build ProteinDocument from sequences and accessions"""
    if is_msa:
        # Process as MSA - remove insertions, convert to upper
        processed_seqs = []
        for seq in sequences:
            # Remove lowercase (insertions) and convert to upper
            processed_seq = ''.join(c for c in seq if not c.islower()).upper()
            processed_seqs.append(processed_seq)
        sequences = processed_seqs
    
    # Representative is first by default if present
    rep = accessions[0] if len(accessions) > 0 else None
    return ProteinDocument(
        sequences=sequences,
        accessions=accessions,
        identifier=identifier,
        representative_accession=rep,
    )


def handler(job):
    """
    RunPod handler for ProFam protein sequence generation.
    
    Expected input:
    {
        "fasta_content": ">seq1\nMKLLILTCLL...\n>seq2\nMKLLVL...",  # Input FASTA/MSA content
        "num_samples": 3,  # Number of sequences to generate
        "sampler": "single",  # "single" or "ensemble"
        "temperature": 1.0,  # Optional, sampling temperature
        "top_p": 0.95,  # Optional, nucleus sampling
        "max_tokens": 8192,  # Optional, max tokens
        "max_generated_length": null,  # Optional, max length of generated sequences
        "num_prompts_in_ensemble": 8,  # For ensemble sampler
        "is_msa": true,  # Whether input is MSA
        "seed": 42  # Random seed
    }
    
    Returns:
    {
        "sequences": ["MKLLI...", "MKVLI..."],
        "accessions": ["generated_0", "generated_1"],
        "scores": [-12.3, -14.5]
    }
    """
    
    try:
        job_input = job.get('input', {})
        
        # Get inputs
        fasta_content = job_input.get('fasta_content')
        num_samples = job_input.get('num_samples', 3)
        sampler_type = job_input.get('sampler', 'single')
        temperature = job_input.get('temperature', None)
        top_p = job_input.get('top_p', 0.95)
        max_tokens = job_input.get('max_tokens', 8192)
        max_generated_length = job_input.get('max_generated_length', None)
        num_prompts_in_ensemble = job_input.get('num_prompts_in_ensemble', 8)
        is_msa = job_input.get('is_msa', True)
        seed = job_input.get('seed', 42)
        continuous_sampling = job_input.get('continuous_sampling', False)
        attn_implementation = job_input.get('attn_implementation', 'sdpa')
        
        # Validate inputs
        if not fasta_content:
            return {"error": "No fasta_content provided"}
        
        # Set seed for reproducibility
        seed_all(seed)
        
        # Device setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Using device: {device}, dtype: {dtype}")
        
        # Load model
        print(f"Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            return {"error": f"Model checkpoint not found at {MODEL_PATH}"}
        
        # Load with attention implementation override
        try:
            ckpt_blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            hyper_params = ckpt_blob.get("hyper_parameters", {})
            cfg_obj = hyper_params.get("config", None)
            if cfg_obj is None:
                # Load without override if config not found
                model = LlamaLitModule.load_from_checkpoint(MODEL_PATH)
            else:
                # Override attention implementation
                setattr(cfg_obj, "attn_implementation", attn_implementation)
                setattr(cfg_obj, "_attn_implementation", attn_implementation)
                model = LlamaLitModule.load_from_checkpoint(MODEL_PATH, config=cfg_obj)
        except Exception as e:
            print(f"Warning: Could not override attention implementation: {e}")
            model = LlamaLitModule.load_from_checkpoint(MODEL_PATH)
        
        model.eval()
        model.to(device, dtype=dtype)
        
        # Parse FASTA content
        accessions, sequences = read_fasta_from_string(fasta_content)
        if len(sequences) == 0:
            return {"error": "No sequences found in FASTA content"}
        
        print(f"Loaded {len(sequences)} sequences from input")
        
        # Build protein document
        pool = build_pool_from_sequences(sequences, accessions, identifier="input", is_msa=is_msa)
        
        # Calculate max generation length
        if max_generated_length is None:
            max_gen_len = int(max(pool.sequence_lengths) * 1.2)
        else:
            max_gen_len = min(max_generated_length, int(max(pool.sequence_lengths) * 1.2))
        
        if continuous_sampling:
            max_gen_len = None
        
        # Setup preprocessor
        doc_token = "[RAW]"
        cfg = AlignedProteinPreprocessingConfig(
            document_token=doc_token,
            defer_sampling=True if sampler_type == "ensemble" else False,
            padding="do_not_pad",
            shuffle_proteins_in_document=True,
            keep_insertions=True,
            to_upper=True,
            keep_gaps=False,
            use_msa_pos=False,
            max_tokens_per_example=None if sampler_type == "ensemble" else max_tokens,
        )
        preprocessor = ProteinDocumentPreprocessor(cfg=cfg)
        
        # Build sampler
        if sampler_type == "ensemble":
            builder = EnsemblePromptBuilder(preprocessor=preprocessor, shuffle=True, seed=seed)
            sampler = ProFamEnsembleSampler(
                name="ensemble_sampler",
                model=model,
                prompt_builder=builder,
                document_token=doc_token,
                reduction="mean_probs",
                temperature=temperature,
                top_p=top_p,
                add_final_sep=True,
            )
        else:
            builder = PromptBuilder(preprocessor=preprocessor, prompt_is_aligned=True, seed=seed)
            sampling_kwargs = {}
            if top_p is not None:
                sampling_kwargs["top_p"] = top_p
            if temperature is not None:
                sampling_kwargs["temperature"] = temperature
            sampler = ProFamSampler(
                name="single_sampler",
                model=model,
                prompt_builder=builder,
                document_token=doc_token,
                sampling_kwargs=sampling_kwargs if len(sampling_kwargs) > 0 else None,
                add_final_sep=True,
            )
        
        sampler.to(device)
        
        # Generate sequences
        print(f"Generating {num_samples} sequences using {sampler_type} sampler...")
        
        if sampler_type == "ensemble":
            generated_sequences, scores, _ = sampler.sample_seqs_ensemble(
                protein_document=pool,
                num_samples=num_samples,
                max_tokens=max_tokens,
                num_prompts_in_ensemble=min(num_prompts_in_ensemble, len(pool.sequences)),
                max_generated_length=max_gen_len,
                continuous_sampling=continuous_sampling,
            )
        else:
            # Adjust max_tokens for single sampler
            if max_gen_len:
                preprocessor.cfg.max_tokens_per_example = max_tokens - max_gen_len
                builder = PromptBuilder(preprocessor=preprocessor, prompt_is_aligned=True)
                sampling_kwargs = {}
                if top_p is not None:
                    sampling_kwargs["top_p"] = top_p
                if temperature is not None:
                    sampling_kwargs["temperature"] = temperature
                sampler = ProFamSampler(
                    name="single_sampler",
                    model=model,
                    prompt_builder=builder,
                    document_token=doc_token,
                    sampling_kwargs=sampling_kwargs if len(sampling_kwargs) > 0 else None,
                    add_final_sep=True,
                )
            
            generated_sequences, scores, _ = sampler.sample_seqs(
                protein_document=pool,
                num_samples=num_samples,
                max_tokens=max_tokens,
                max_generated_length=max_gen_len,
                continuous_sampling=continuous_sampling,
            )
        
        # Format output
        generated_accessions = [f"generated_{i}_score_{score:.3f}" for i, score in enumerate(scores)]
        
        print(f"Successfully generated {len(generated_sequences)} sequences")
        
        return {
            "sequences": generated_sequences,
            "accessions": generated_accessions,
            "scores": [float(score) for score in scores]  # Convert to Python float
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in handler: {error_trace}")
        return {
            "error": f"Failed to process: {str(e)}",
            "traceback": error_trace
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
