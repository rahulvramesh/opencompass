# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenCompass is a comprehensive platform for evaluating large language models (LLMs). It provides a flexible, distributed evaluation framework supporting 100+ datasets and 50+ model types, with both local and cluster execution capabilities.

## Common Commands

### Installation

```bash
# Install from source for development
pip install -e .

# Install with specific backend support
pip install -e ".[vllm]"      # vLLM acceleration
pip install -e ".[lmdeploy]"  # LMDeploy acceleration
pip install -e ".[api]"       # API model support
pip install -e ".[full]"      # All optional dependencies
```

### Running Evaluations

```bash
# Simple CLI evaluation
opencompass --models hf_internlm2_5_1_8b_chat --datasets demo_gsm8k_chat_gen

# Using Python config files
opencompass examples/eval_chat_demo.py

# With accelerator backend
opencompass --models hf_internlm2_5_1_8b_chat --datasets demo_gsm8k_chat_gen -a lmdeploy

# Data parallel (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1 opencompass --datasets demo_gsm8k_chat_gen --hf-path internlm/internlm2_5-1_8b-chat --max-num-worker 2

# Run only specific phase
opencompass examples/eval_chat_demo.py --mode infer  # Only inference
opencompass examples/eval_chat_demo.py --mode eval   # Only evaluation
opencompass examples/eval_chat_demo.py --mode viz    # Only summarization

# Reuse existing results (incremental evaluation)
opencompass examples/eval_chat_demo.py --reuse latest
```

### Cluster Execution

```bash
# SLURM cluster
opencompass examples/eval_chat_demo.py --slurm --partition gpu_partition

# Alibaba DLC
opencompass examples/eval_chat_demo.py --dlc
```

### Utilities

```bash
# List available models and datasets
python tools/list_configs.py
python tools/list_configs.py llama mmlu  # Filter by keyword

# Test API models
python tools/test_api_model.py

# View prompts for debugging
python tools/prompt_viewer.py
```

## Architecture Overview

OpenCompass follows a three-phase evaluation pipeline: **Inference → Evaluation → Summarization**. Each phase uses a Partitioner-Runner pattern for distributed execution.

### Core Components

#### 1. Entry Point (opencompass/cli/main.py)
- Main CLI that orchestrates the entire evaluation workflow
- Handles config loading from files or CLI arguments
- Manages three execution phases with mode flag (`--mode all|infer|eval|viz`)

#### 2. Models (opencompass/models/)
- **Base**: `BaseModel` defines interface (`generate()`, `get_ppl()`)
- **Local Models**: `HuggingFace`, `VLLMwithChatTemplate`, `TurboMindModelwithChatTemplate`
- **API Models**: `OpenAI_API`, `Claude_API`, and 20+ other API providers
- **Registry**: All models registered via `@MODELS.register_module()` decorator
- Built via `build_model_from_cfg()` which handles config preprocessing

#### 3. Datasets (opencompass/datasets/)
- 100+ dataset implementations (MMLU, GSM8K, HumanEval, etc.)
- Each dataset config specifies:
  - `reader_cfg`: How to load data (input_columns, output_column, splits)
  - `infer_cfg`: How to run inference (prompt templates, retrievers, inferencers)
  - `eval_cfg`: How to evaluate (evaluators, postprocessors)
- Loaded via `build_dataset_from_cfg()` with lazy loading

#### 4. Tasks (opencompass/tasks/)
- **OpenICLInferTask**: Runs model inference on dataset, saves predictions to `predictions/{model}/{dataset}.json`
- **OpenICLEvalTask**: Evaluates predictions, saves results to `results/{model}/{dataset}.json`
- Each task generates shell commands with proper GPU allocation
- Supports distributed execution via `torch.distributed.run`

#### 5. Partitioners (opencompass/partitioners/)
- Transform config into independent task list: `(models, datasets) → List[Task]`
- **NaivePartitioner**: Groups n model-dataset pairs per task
- **SizePartitioner**: Groups by dataset size for balanced workload
- **NumWorkerPartitioner**: Partitions by number of workers
- Checks for existing outputs and skips completed work

#### 6. Runners (opencompass/runners/)
- **LocalRunner**: ThreadPoolExecutor with GPU allocation (respects `CUDA_VISIBLE_DEVICES`)
- **SlurmRunner**: Submits to SLURM via `srun`
- **DLCRunner**: Alibaba cloud execution
- Pattern: `runner.launch(tasks)` returns status, `runner.summarize()` reports failures

#### 7. Summarizers (opencompass/summarizers/)
- **DefaultSummarizer**: Aggregates results from `results/` directory
- Supports summary groups (e.g., MMLU subsets with weighted averaging)
- Outputs tables to `summary/` directory

#### 8. OpenICL Framework (opencompass/openicl/)
- Handles in-context learning (few-shot prompting)
- **Retrievers**: Select few-shot examples (`ZeroRetriever`, `BM25Retriever`)
- **Inferencers**: Execute inference (`GenInferencer`, `PPLInferencer`, `ChatInferencer`)
- **Prompt Templates**: Format prompts with roles and variables

### Evaluation Flow

```
User Command
    ↓
CLI parses args and loads config
    ↓
INFERENCE PHASE:
├── Partitioner creates inference tasks (model-dataset pairs)
├── Runner executes tasks (local/SLURM/DLC)
└── For each task:
    ├── Build model from config
    ├── Build dataset and ICL components
    ├── Generate predictions via model.generate()
    └── Save to predictions/{model}/{dataset}.json
    ↓
EVALUATION PHASE:
├── Partitioner creates evaluation tasks
├── Runner executes tasks
└── For each task:
    ├── Load predictions and ground truth
    ├── Post-process outputs
    ├── Compute metrics
    └── Save to results/{model}/{dataset}.json
    ↓
SUMMARIZATION PHASE:
├── Summarizer loads all results
├── Aggregates metrics and computes group averages
└── Outputs summary tables to summary/
```

## Configuration System

Configs are Python files using MMEngine's config system:

```python
from mmengine.config import read_base

with read_base():
    # Import base configs
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import models

# Compose your evaluation
datasets = mmlu_datasets
models = models
```

### Config Structure

```python
# Top-level keys
models = [...]          # List of model configs
datasets = [...]        # List of dataset configs

# Optional: Advanced task distribution
model_dataset_combinations = [
    {'models': [modelA], 'datasets': [dataset1, dataset2]},
    {'models': [modelB], 'datasets': [dataset3]}
]

# Execution configuration
infer = dict(
    partitioner=dict(type='SizePartitioner', ...),
    runner=dict(type='LocalRunner', max_num_workers=8)
)

eval = dict(
    partitioner=dict(type='SizePartitioner', ...),
    runner=dict(type='LocalRunner', max_num_workers=8)
)
```

### Model Config Keys

- `type`: Model class name (e.g., 'HuggingFacewithChatTemplate')
- `path`: Model path (HuggingFace ID or local path)
- `abbr`: Short name for output files
- `max_seq_len`: Maximum sequence length
- `meta_template`: Chat template configuration
- `generation_kwargs`: Sampling parameters (temperature, top_p, max_new_tokens)
- `batch_size`: Batch size for inference
- `run_cfg`: Task execution settings (num_gpus, num_procs)

### Dataset Config Keys

- `abbr`: Short name for dataset
- `type`: Dataset class
- `path`: Data source (HuggingFace/local/ModelScope)
- `reader_cfg`: Data loading (input_columns, output_column, splits)
- `infer_cfg`: Inference settings (prompt_template, retriever, inferencer)
- `eval_cfg`: Evaluation settings (evaluator, postprocessors)

## Key Directories

- `opencompass/cli/`: CLI entry point
- `opencompass/models/`: Model implementations
- `opencompass/datasets/`: Dataset classes
- `opencompass/configs/`: Pre-configured models and datasets
- `opencompass/tasks/`: Task execution (inference, evaluation)
- `opencompass/runners/`: Execution backends (local, SLURM, DLC)
- `opencompass/partitioners/`: Task distribution strategies
- `opencompass/summarizers/`: Result aggregation
- `opencompass/openicl/`: In-context learning framework
- `opencompass/utils/`: Shared utilities (build, logging, postprocessors)
- `examples/`: Example evaluation scripts
- `tools/`: Utility scripts

## Registry System

OpenCompass uses MMEngine's registry for component discovery:

```python
from opencompass.registry import MODELS, DATASETS, RUNNERS, PARTITIONERS

# Register a component
@MODELS.register_module()
class MyCustomModel(BaseModel):
    ...

# Build from config
model = MODELS.build(model_cfg)
```

Key registries:
- `MODELS`: Model implementations
- `DATASETS`: Dataset classes
- `TASKS`: Task types (infer, eval)
- `RUNNERS`: Execution backends
- `PARTITIONERS`: Task distribution strategies
- `ICL_INFERENCERS`: In-context learning inferencers
- `ICL_RETRIEVERS`: Few-shot example retrievers
- `ICL_EVALUATORS`: Evaluation metrics

## Working with Models

### Adding a New HuggingFace Model

Create a config file in `opencompass/configs/models/`:

```python
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='my_model_hf',
        path='org/model-name',
        max_seq_len=4096,
        tokenizer_path='org/model-name',
        tokenizer_kwargs=dict(trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        generation_kwargs=dict(do_sample=True, temperature=0.7),
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
```

### Using Accelerators

```bash
# Transform HuggingFace config to vLLM
opencompass --models hf_model --datasets mmlu -a vllm

# Transform to LMDeploy
opencompass --models hf_model --datasets mmlu -a lmdeploy
```

The `--accelerator` flag automatically transforms model configs to use accelerated backends.

## Working with Datasets

### Adding a New Dataset

1. Create dataset class in `opencompass/datasets/`:

```python
from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET

@LOAD_DATASET.register_module()
class MyDataset(BaseDataset):
    @staticmethod
    def load(**kwargs):
        # Load and return dataset
        return dataset
```

2. Create config in `opencompass/configs/datasets/`:

```python
from opencompass.datasets import MyDataset
from opencompass.openicl import GenInferencer, ZeroRetriever

my_dataset = [
    dict(
        abbr='my_dataset',
        type=MyDataset,
        path='path/to/data',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='answer',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='Question: {question}\nAnswer:',
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer', max_out_len=512),
        ),
        eval_cfg=dict(
            evaluator=dict(type='Accuracy'),
        ),
    )
]
```

## Output Structure

After evaluation, the work directory contains:

```
outputs/default/{timestamp}/
├── configs/              # Saved config for reproducibility
├── logs/                 # Execution logs
│   ├── infer/           # Inference logs
│   └── eval/            # Evaluation logs
├── predictions/         # Model predictions
│   └── {model}/
│       └── {dataset}.json
├── results/             # Evaluation results
│   └── {model}/
│       └── {dataset}.json
└── summary/             # Aggregated results
    └── summary_{timestamp}.csv
```

## Important Implementation Details

### Prediction File Format
```json
{
    "prediction": ["answer1", "answer2", ...],
    "gold": ["ground_truth1", "ground_truth2", ...]
}
```

### Result File Format
```json
{
    "metric_name": metric_value,
    "details": [
        {"pred": "...", "gold": "...", "correct": true},
        ...
    ]
}
```

### Task Naming Convention
- Tasks are named: `{name_prefix}/{model_abbr}/{dataset_abbr}`
- Example: `OpenICLInfer/qwen2_1_5b_instruct/gsm8k`

### GPU Allocation Strategy (LocalRunner)
1. Parse `CUDA_VISIBLE_DEVICES` to get available GPUs
2. Respect `max_workers_per_gpu` (default: 1)
3. Respect `max_num_workers` (global limit)
4. Assign GPUs round-robin to tasks
5. Set `CUDA_VISIBLE_DEVICES` for each task subprocess

### Reuse Mechanism
- `--reuse latest`: Skip completed tasks (checks for existing output files)
- Partitioner calls `get_infer_output_path()` and `get_eval_output_path()`
- Enables incremental evaluation when adding new models/datasets

## Development Tips

### Adding Custom Evaluators

Extend `BaseEvaluator` or use built-in evaluators:

```python
from opencompass.openicl.icl_evaluator import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        # Compute metric
        return {'my_metric': score}
```

### Debugging Prompts

Use the prompt viewer to inspect actual prompts sent to models:

```bash
python tools/prompt_viewer.py
```

### Custom Post-processors

Add text post-processors in `opencompass/utils/text_postprocessors.py`:

```python
from opencompass.registry import TEXT_POSTPROCESSORS

@TEXT_POSTPROCESSORS.register_module()
def my_postprocessor(text: str) -> str:
    # Extract/clean text
    return cleaned_text
```

## Testing

Run tests from the repository root:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_dataset.py
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU selection
- `DATASET_SOURCE`: Set to "ModelScope" to use ModelScope datasets
- `HF_ENDPOINT`: HuggingFace mirror (e.g., "https://hf-mirror.com")
- `OPENAI_API_KEY`: For OpenAI API models
- API keys for other providers (varies by model)

## Common Issues

### Out of Memory
- Reduce `batch_size` in model config
- Use `--accelerator vllm` or `--accelerator lmdeploy`
- Enable model parallelism: `model_kwargs=dict(device_map='auto')`

### Slow Evaluation
- Increase `max_num_workers` for more parallel tasks
- Use `--accelerator` for faster inference
- Use `SizePartitioner` for better load balancing

### Task Failures
- Check logs in `logs/infer/` and `logs/eval/`
- Use `--debug` flag for verbose output
- Verify GPU availability with `nvidia-smi`
