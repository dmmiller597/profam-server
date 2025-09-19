# ProFam Server - Protein Sequence Generation on RunPod

This is a serverless deployment of the **ProFam** protein language model for generating protein sequences based on input protein families or MSAs (Multiple Sequence Alignments).

## What is ProFam?

ProFam is a protein language model that can generate new protein sequences that are similar to input protein families. It's trained on protein sequence data and can:
- Generate novel protein sequences from MSA inputs
- Maintain family-specific characteristics
- Produce sequences with likelihood scores

## Quick Start

### Building the Docker Image

```bash
docker build -f Dockerfile.profam -t profam-runpod .
```

### Testing Locally

```bash
docker run --gpus all -p 8000:8000 profam-runpod
```

### Deploying to RunPod

1. Push your image to a container registry (e.g., Docker Hub)
2. Create a new serverless endpoint on RunPod
3. Configure with your container image

## API Usage

### Input Format

Send a POST request with JSON payload:

```json
{
  "input": {
    "fasta_content": ">protein1\nMKLLILTCLL...\n>protein2\nMKLLVL...",
    "num_samples": 3,
    "sampler": "single",
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 8192,
    "is_msa": true,
    "seed": 42
  }
}
```

### Parameters

- **fasta_content** (required): FASTA/MSA format protein sequences
- **num_samples**: Number of sequences to generate (default: 3)
- **sampler**: "single" or "ensemble" (default: "single")
- **temperature**: Sampling temperature (optional, higher = more diverse)
- **top_p**: Nucleus sampling cutoff (default: 0.95)
- **max_tokens**: Maximum context tokens (default: 8192)
- **max_generated_length**: Max length of generated sequences (optional)
- **is_msa**: Whether input is MSA format (default: true)
- **seed**: Random seed for reproducibility (default: 42)
- **continuous_sampling**: Generate until token budget (default: false)
- **attn_implementation**: Attention type - "sdpa", "flash_attention_2", or "eager" (default: "sdpa")

### Output Format

```json
{
  "sequences": [
    "MKLLILTCLLVAVALANPQEAG...",
    "MKVLILTCLLVAVALANPQDAG..."
  ],
  "accessions": [
    "generated_0_score_-12.345",
    "generated_1_score_-14.567"
  ],
  "scores": [-12.345, -14.567]
}
```

## Example Usage with Python

```python
import requests
import json

# RunPod endpoint URL
url = "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync"
headers = {
    "Authorization": "Bearer {YOUR_API_KEY}",
    "Content-Type": "application/json"
}

# Example MSA input
fasta_content = """
>seq1
MKLLILTCLLVAVALANPQEAG
>seq2
MKVLILTCLLVAVALANPQDAG
>seq3
MKLLVLTCLLVAVALANPQEAG
"""

payload = {
    "input": {
        "fasta_content": fasta_content,
        "num_samples": 5,
        "sampler": "single",
        "temperature": 0.8,
        "is_msa": True
    }
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()

# Extract generated sequences
if "output" in result:
    sequences = result["output"]["sequences"]
    scores = result["output"]["scores"]
    for seq, score in zip(sequences, scores):
        print(f"Score: {score:.3f}")
        print(f"Sequence: {seq}\n")
```

## Sampler Types

### Single Sampler
- Faster, uses single prompt
- Good for quick generation
- Lower memory usage

### Ensemble Sampler
- Uses multiple prompts with different shuffles
- Better quality/diversity
- Higher computational cost
- Set `num_prompts_in_ensemble` to control ensemble size

## Model Details

- **Model**: ProFam checkpoint from [HuggingFace](https://huggingface.co/judewells/pf/blob/main/checkpoints/last.ckpt)
- **Architecture**: Based on LLaMA architecture adapted for proteins
- **Training**: Trained on protein sequence databases
- **Size**: ~1.51 GB

## Performance Tips

1. **Use GPU**: Always deploy with GPU for reasonable performance
2. **Batch Size**: Adjust based on GPU memory
3. **Attention Implementation**: 
   - Use "flash_attention_2" for best performance (if supported)
   - "sdpa" is a good default
   - "eager" for debugging
4. **Temperature**: Lower values (0.7-0.9) for conservative sequences, higher (1.0-1.2) for more diversity

## Troubleshooting

### Out of Memory
- Reduce `num_samples`
- Reduce `max_tokens`
- Use smaller `num_prompts_in_ensemble` for ensemble sampler

### Slow Generation
- Switch from "ensemble" to "single" sampler
- Use "flash_attention_2" if available
- Reduce `max_generated_length`

### Poor Quality Sequences
- Try ensemble sampler
- Adjust temperature
- Provide more diverse input sequences

## References

- [ProFam GitHub](https://github.com/alex-hh/profam)
- [Model Checkpoint](https://huggingface.co/judewells/pf)
- [RunPod Documentation](https://docs.runpod.io/)
