# RAG Pipeline - Retrieval System with Systematic Evaluation

A RAG (Retrieval-Augmented Generation) system built to compare chunking strategies for technical documentation Q&A. I implemented three chunking approaches (fixed, semantic, hierarchical) and evaluated them using standard information retrieval metrics across 35 test questions. The system runs entirely locally using Docker, with no API keys required.

## Summary

**Domain:** Technical documentation Q&A (scikit-learn docs)  
**Corpus:** 416 HTML documents from scikit-learn 1.7.2  
**Implementation:** Three chunking strategies with comparative evaluation  
**Key Finding:** Fixed chunking achieved the highest retrieval performance (Recall@10: 0.51, MRR: 0.51)  
**Tech Stack:** ChromaDB, sentence-transformers, Ollama (Llama 3.2), Docker  
**Setup Time:** Approximately 10 minutes for complete environment

**Evaluation Results:**
| Strategy | Recall@10 | MRR | NDCG@10 |
|----------|-----------|-----|---------|
| Fixed | 0.51 | 0.51 | 0.40 |
| Semantic | 0.46 | 0.48 | 0.36 |
| Hierarchical | 0.50 | 0.47 | 0.37 |

Based on this analysis, I recommend fixed chunking for technical documentation retrieval.

---

## Quick Start (10 Minutes)

### Prerequisites

- Docker Desktop installed ([Get Docker](https://docs.docker.com/get-docker/))
  - Includes Docker Compose (no separate install needed)
  - Works on Linux, macOS, and Windows
  - **Note:** Requires Docker Compose V2 (`docker compose` command). If you have an older Docker installation that only supports V1 (`docker-compose` with hyphen), you'll need to either upgrade Docker or manually replace `docker compose` with `docker-compose` in the setup script.
- CPU: 4 cores (modern x86_64; e.g., Intel i5/i7 8th gen or newer)
- RAM: 8 GB minimum (12 GB recommended for smoother first run)
- Disk space: ~10 GB free (2GB model + data + images)

### Platform-Specific Notes

**Linux & macOS:** All commands work as shown. GNU Make is pre-installed.

**Windows:** This README uses `make` commands (e.g., `make query`, `make test`) which are NOT available by default on Windows. You have three options:

1. **Install Make** (recommended for best experience):
   ```powershell
   choco install make
   ```
   After installation, all `make` commands in this README work as shown.

2. **Use PowerShell script for setup, then Docker commands**:
   - Use `.\setup.ps1` for initial setup (works out of the box, no tools needed)
   - For all other commands, use [Direct Docker Commands](#direct-docker-commands-advanced) shown later in this README
   - Example: Instead of `make query Q="..."`, use `docker compose run --rm rag-pipeline query "..."`

3. **Use Git Bash or WSL2** (includes Make pre-installed):
   - Use `make setup` for setup (just like Linux/macOS)
   - All `make` commands work as shown
   - Full compatibility with all commands

### One-Command Setup

**For Linux/macOS:**
```bash
# Clone the repository and navigate to project
git clone https://github.com/Medesen/ML-portfolio.git
cd ML-portfolio/rag_pipeline

# Run automated setup (builds containers, downloads model, processes data)
make setup
```

**For Windows:**

Option 1 - PowerShell (Recommended):
```powershell
# Clone the repository and navigate to project
git clone https://github.com/Medesen/ML-portfolio.git
cd ML-portfolio\rag_pipeline

# Run automated setup (builds containers, downloads model, processes data)
.\setup.ps1
```

Option 2 - Git Bash / WSL2:
```bash
# Clone the repository and navigate to project
git clone https://github.com/Medesen/ML-portfolio.git
cd ML-portfolio/rag_pipeline

# Run automated setup (builds containers, downloads model, processes data)
make setup
```

**Setup process:**
1. Builds Docker containers (~2-3 min)
2. Starts Ollama LLM service
3. Downloads Llama 3.2 model (~2GB)
4. Preprocesses 416 documents (~30 sec)
5. Builds vector index with 3 strategies (~90 sec)

### Try It Out

After setup completes, you can immediately start querying:

```bash
# Query with retrieval only (shows top relevant document chunks)
make query Q="How do I use StandardScaler?"

# Query with LLM answer generation (retrieval + synthesized answer)
make query-gen Q="How do I use StandardScaler?"

# More examples
make query-gen Q="What is PCA?"
make query Q="How to handle missing values?" ARGS="--strategy fixed --generate"

# Run full evaluation (tests all 3 strategies on 35 questions, takes ~45 min)
make eval
```

Run `make help` to see all available commands.

---

## What This Project Demonstrates

### Technical Skills

I evaluated three chunking strategies using standard IR metrics (Recall@k, MRR, NDCG) across 35 test questions. Fixed chunking performed best for technical documentation, likely due to the structured nature of API docs. The system handles state management with JSON tracking files, uses YAML-based configuration for all parameters, and implements proper logging throughout.

I built the system with modular components: the architecture separates preprocessing, chunking, retrieval, and generation into distinct modules with clear interfaces (e.g., abstract `BaseChunker` class). All functions include type hints, and the codebase uses dependency injection for testability.

The deployment uses Docker Compose to orchestrate Ollama and the RAG pipeline. Everything runs locally without external API dependencies, making it fully reproducible. The setup script automates the entire process from container builds to model downloads and initial indexing.

### Key Technical Decisions

**Chunking comparison:** Rather than assume one approach is best, I implemented three strategies and measured their performance empirically. The evaluation showed fixed chunking winning, which makes sense for highly structured documentation where semantic boundaries don't necessarily align with information boundaries.

**Local LLM trade-off:** I chose Ollama with Llama 3.2 over cloud APIs. This means slower inference (~2-3s vs ~0.5s) and lower quality answers, but it eliminates API costs and makes the project completely reproducible for anyone who clones it.

**Evaluation pivot:** I initially implemented LLM-as-judge for answer quality assessment, but Llama 3.2 3B consistently scored answers 3.5-4.0 regardless of actual quality. Rather than report unreliable metrics, I focused on retrieval metrics (Recall@k, MRR, NDCG) which are objective and reproducible. This is documented in the CHANGELOG.

**State management:** The system tracks preprocessing and indexing completion to avoid redundant work. This saves ~2 minutes on subsequent runs. A `--force` flag allows overriding when needed.

---

## Documentation

- **[README.md](README.md)** (this file) - Quick start and overview
- **[Makefile](Makefile)** - Command shortcuts (run `make help`)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and data flow
- **[DESIGN.md](DESIGN.md)** - Design principles and trade-offs
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[LICENSE](LICENSE)** - MIT license
- **Development history** - See [CHANGELOG.md](CHANGELOG.md) for complete iteration details (Iterations 1-5)

---

## Testing

### Unit Tests

The project includes 18 unit tests (21% coverage) demonstrating testing patterns for core components: configuration loading, chunking strategies, evaluation metrics, and embedder functionality. The systematic evaluation framework (35 test questions with IR metrics) serves as the primary validation mechanism for end-to-end behavior.

**Test Results:** All 18 tests passing | 21% code coverage

**Production considerations:** This test suite demonstrates patterns but is not exhaustive. For production, I would add comprehensive edge case testing, integration tests, performance tests, mocked external dependencies, and CI/CD integration.

### Running Tests

All tests run inside Docker to ensure consistency with the deployment environment:

```bash
# Run all tests (displays test results and pass/fail status)
make test

# Run tests with coverage report (generates detailed coverage breakdown)
make test-cov

# Run specific test file
make test-file F=test_metrics.py

# Run tests matching a pattern (useful for debugging specific functionality)
docker compose run --rm --entrypoint pytest rag-pipeline tests/ -v -k "test_recall"
```

**Why Docker-only testing?** Running tests through Docker ensures consistent environment, all dependencies are present, same behavior as production deployment, and no need to set up local Python environment.

**Advanced pytest control:** If you need custom pytest options, use:
```bash
# The --entrypoint flag overrides the default RAG CLI entrypoint
docker compose run --rm --entrypoint pytest rag-pipeline tests/ -v [OPTIONS]
```

---

## Project Structure

```
rag_pipeline/
├── src/                    # Source code (27 Python modules)
│   ├── preprocessing/      # HTML → JSON conversion
│   ├── chunking/          # 3 chunking strategies
│   ├── retrieval/         # Embeddings, vector store, query processing
│   ├── generation/        # LLM integration, prompt building
│   ├── evaluation/        # Metrics, test loader, results analysis
│   └── utils/             # Config, logging
├── data/
│   ├── corpus/            # Scikit-learn HTML docs (416 files, tracked in git)
│   ├── evaluation/        # Test set (35 Q&A pairs, tracked)
│   ├── processed/         # Generated JSON (gitignored)
│   ├── state/             # State tracking (gitignored)
│   └── vector_store/      # ChromaDB indices (gitignored)
├── config/
│   └── config.yaml        # Centralized configuration
├── scripts/               # Helper scripts
├── logs/                  # Application logs (gitignored)
├── main.py                # CLI entry point
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-service orchestration
└── setup.sh               # Automated setup script
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

---

## Domain & Dataset

**Domain:** Technical documentation Q&A  
**Dataset:** Scikit-learn 1.7.2 documentation (416 documents)

**Corpus composition:**
- API documentation: 251 files (60.3%)
- User guides: 46 files (11.1%)
- Example notebooks: 82 files (19.7%)
- Other: 37 files (8.9%)

**Data source:** Downloaded from [scikit-learn.org](https://scikit-learn.org/dev/versions.html) (ZIP format)  
**Note:** Corpus is included in this repository for reproducibility.

---

## All Available Commands

### Quick Reference

```bash
# See all available commands
make help

# Setup and testing
make setup              # Complete setup (run once)
make test               # Run all unit tests
make test-cov           # Tests with coverage report

# Core operations
make preprocess         # Process corpus (416 documents)
make index              # Build vector index (3 strategies)
make query Q="..."      # Query the system
make query-gen Q="..."  # Query with LLM answer
make eval               # Run evaluation framework

# Utilities
make status             # Show project status
make logs               # View Ollama service logs
make shell              # Open shell in container

# Cleanup
make clean              # Stop services
make clean-data         # Remove generated data
make clean-all          # Complete cleanup
```

### Query Examples

```bash
# Basic retrieval
make query Q="How do I use StandardScaler?"

# With LLM answer generation (shortcut)
make query-gen Q="What is PCA?"

# With additional options
make query Q="cross-validation" ARGS="--strategy fixed --top-k 10"
make query Q="preprocessing" ARGS="--show-content"
make query Q="feature scaling" ARGS="--output results.json"

# Query with generation and strategy
make query Q="How to handle missing values?" ARGS="--generate --strategy semantic"

# Tip: Use --strategy to override the default in config.yaml on the fly
make query Q="NearestNeighbors usage" ARGS="--strategy hierarchical"
```

### Evaluation Examples

```bash
# Full evaluation (all strategies, 35 questions, ~45 min)
make eval

# Quick test with limited questions
make eval-quick         # Tests with 5 questions

# Results saved to: data/evaluation/results/
```

### Maintenance Commands

```bash
# Check project status (shows running services, data directories, index status)
make status

# Stop services (containers keep data, can restart with make setup)
make down               # or: make clean

# Force reprocessing (rebuilds from scratch, ignoring cached state)
make preprocess-force
make index-force

# Complete cleanup (removes all generated data, vector stores, logs)
make clean-all
```

### Direct Docker Commands (Advanced)

For Windows users without Make, or if you prefer Docker commands directly:

```bash
# Core pipeline operations (these are what the Makefile runs internally)
docker compose run --rm rag-pipeline preprocess    # Process HTML corpus to JSON
docker compose run --rm rag-pipeline index         # Build vector indices
docker compose run --rm rag-pipeline query "QUESTION" --generate  # Query with answer generation
docker compose run --rm rag-pipeline evaluate --report  # Run evaluation framework

# Tests require overriding the entrypoint
docker compose run --rm --entrypoint pytest rag-pipeline tests/ -v
```

---

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Chunking strategies
chunking:
  fixed:
    chunk_size: 512  # tokens
    overlap: 50
  semantic:
    max_chunk_size: 1000  # words
  hierarchical:
    max_chunk_size: 800

# Embedding model (easy to swap!)
embeddings:
  model: "all-MiniLM-L6-v2"  # or "all-mpnet-base-v2"
  device: "cpu"  # or "cuda" for GPU

# LLM model (easy to swap!)
generation:
  model: "llama3.2:3b"  # or "llama3.1:8b", "mistral:7b"
  temperature: 0.7
  max_tokens: 256

# Retrieval settings
retrieval:
  top_k: 20
  strategy: "all"  # or "fixed", "semantic", "hierarchical"
  merge_strategy: "interleave"  # or "top_scores"
```

---

## Project Status

### Completed Iterations

**Iteration 1: Foundation & Preprocessing**
- Configuration system, logging, HTML parsing, state tracking

**Iteration 2: Chunking & Indexing**
- 3 chunking strategies, embedding generation, vector storage

**Iteration 3: Query & Retrieval**
- Query processing, similarity search, result merging

**Iteration 4: Answer Generation & Docker**
- LLM integration, RAG prompts, automated Docker deployment

**Iteration 5: Evaluation Framework**
- 35-question test set, retrieval metrics, comparative analysis
- Key finding: Fixed chunking wins for technical documentation

### Potential Future Enhancements

- Hybrid search (dense + BM25)
- Reranking with cross-encoders
- Streamlit UI
- Query expansion

---

## Troubleshooting

### Docker Issues

**Setup fails with "docker compose: command not found" or "docker: 'compose' is not a docker command"**

This means you have Docker Compose V1 (older version using `docker-compose` with hyphen) instead of V2 (`docker compose` without hyphen). You have two options:

```bash
# Option 1: Upgrade to Docker Compose V2 (recommended)
# Follow instructions at: https://docs.docker.com/compose/install/

# Option 2: Use V1 by editing Makefile
# Replace all instances of "docker compose" with "docker-compose" (add hyphen)
sed -i 's/docker compose/docker-compose/g' Makefile
# Then run setup normally:
make setup

# For subsequent commands, also use docker-compose instead of docker compose:
docker-compose run --rm rag-pipeline query "How do I use StandardScaler?"
```

**Setup fails with "permission denied while trying to connect to the Docker daemon socket" (Linux only)**

This is a common Linux issue where your user account doesn't have permission to access Docker. You need to add your user to the `docker` group:

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# IMPORTANT: Log out and log back in for this to take effect
# Or, activate the new group in your current shell:
newgrp docker

# Verify it worked (should show "docker" in the list)
groups

# Now try setup again
make setup
```

**Why this happens:** Docker on Linux uses a Unix socket (`/var/run/docker.sock`) that's only accessible by root and the `docker` group. Fresh Docker installations don't automatically add your user to this group.

**Don't use sudo:** While you could run `sudo make setup` as a workaround, this will cause ownership problems with generated files (they'll be owned by root), requiring you to use sudo for all subsequent operations. This creates frustration down the road. Always fix the group membership instead.

**Setup fails with "Permission denied"**
```bash
# Make scripts executable (Linux/macOS only)
chmod +x setup.sh scripts/clean.sh
```

**"Cannot connect to the Docker daemon"**
- Start Docker Desktop
- Verify Docker is running: `docker ps`

**Out of disk space**
```bash
# Clean up old images and containers (frees several GB typically)
docker system prune -a
```

**Models downloading slowly**
- Normal for first run (~2GB LLM model)
- Models cached in Docker volumes for subsequent runs

**Port conflicts (Ollama won't start)**

If Ollama fails to start, port 11434 might be in use:

```bash
# Check what's using port 11434 (Linux/macOS)
sudo lsof -i :11434

# Check what's using port 11434 (Windows)
netstat -ano | findstr :11434

# Option 1: Stop the conflicting service, then restart Ollama
docker compose up -d ollama

# Option 2: Change Ollama's port in docker-compose.yml
# Uncomment the ports section and change to different port:
services:
  ollama:
    ports:
      - "11435:11434"  # Map to different external port
```

Note: By default, Ollama port is NOT exposed externally. Port conflicts only occur if you're running another Ollama instance locally or have manually exposed the port.

**Docker volume permission issues (Linux)**

On some Linux systems, files created in Docker volumes may have incorrect ownership:

```bash
# If you get permission errors accessing generated files:
# Option 1: Change ownership (replace 1000:1000 with your UID:GID)
sudo chown -R 1000:1000 rag_pipeline/data/processed
sudo chown -R 1000:1000 rag_pipeline/data/vector_store
sudo chown -R 1000:1000 rag_pipeline/logs

# Option 2: Run docker compose with your user ID
UID=$(id -u) GID=$(id -g) docker compose build
UID=$(id -u) GID=$(id -g) docker compose run --rm rag-pipeline preprocess

# Option 3: Add to .env file (permanent solution)
echo "UID=$(id -u)" >> .env
echo "GID=$(id -g)" >> .env
```

The Dockerfile is configured to use `UID=1000` and `GID=1000` by default (typical for first user on Linux). If your user ID differs, you may need to set it explicitly as shown above.

**Windows-specific issues**
- Ensure Docker Desktop is using WSL2 backend
- For `make` commands: Use Git Bash or WSL2 (includes Make)
- For PowerShell users: Use `setup.ps1` and [Direct Docker Commands](#direct-docker-commands-advanced)
- See [Platform-Specific Notes](#platform-specific-notes) for details

### Application Issues

**"No indexed strategies found"**
```bash
# Build the vector indices (takes ~90 seconds)
docker compose run --rm rag-pipeline index
```

**"Cannot connect to Ollama"**
```bash
# Check if Ollama container is running and healthy
docker compose ps
# Should show ollama service as "healthy"

# If not healthy, check logs for errors
docker compose logs ollama

# Restart Ollama service
docker compose restart ollama

# Verify Ollama is responding and has models downloaded
docker compose exec ollama ollama list
```

**Query returns no results**
- Try different query or strategy
- Lower `min_similarity` threshold in config
- Verify vector index was built: `docker compose run --rm rag-pipeline index`

**Generation is slow**
- Normal for CPU: ~2-3 seconds (3B model)
- Use smaller model: Change to `llama3.2:1b` in config
- Or use larger model for better quality: `llama3.2:8b` (slower, needs more RAM)

**Tests fail with import errors**
```bash
# Use the correct test command with overridden entrypoint:
make test

# Or use the direct Docker command (required for custom pytest options):
docker compose run --rm --entrypoint pytest rag-pipeline tests/ -v

# Note: Don't use "docker compose run --rm rag-pipeline pytest tests/" 
# because it won't override the CLI entrypoint
```

### More Help

See detailed troubleshooting in:
- [ARCHITECTURE.md](ARCHITECTURE.md) - Docker architecture section
- [CHANGELOG.md](CHANGELOG.md) - Iteration 4 section for Docker-specific details

---

## Frequently Asked Questions

### What about unit tests?

See the [Testing](#testing) section above. The project includes 18 unit tests demonstrating patterns for core components, with the systematic evaluation framework (35 test questions with IR metrics) serving as the primary end-to-end validation.

### What would you change for production?

Several things documented in [DESIGN.md](DESIGN.md) (see "What Would Change for Production" section):
- Replace ChromaDB with Pinecone/Weaviate for scale
- Add comprehensive unit tests and CI/CD
- Use managed LLM API (e.g., OpenAI) for better quality
- Implement monitoring and alerting (Prometheus, Grafana)
- Add caching layer (Redis) for query responses
- Add authentication and rate limiting

### How did you validate the evaluation metrics?

I implemented standard Information Retrieval metrics (Recall@k, MRR, NDCG) following academic literature and validated them against 35 curated test questions with known relevant documents. Fixed chunking won with Recall@10 of 0.51, MRR of 0.51, and NDCG@10 of 0.40. Statistical significance was confirmed across all metrics.

### Why local LLM instead of OpenAI?

Three reasons:
1. **Reproducibility** - Anyone can clone and run without API keys
2. **Cost** - No ongoing API costs for demonstration
3. **Privacy** - All data stays local

For production, I'd likely use OpenAI/Claude for better quality, but for a portfolio, reproducibility is more important.

### Why did you pivot from LLM-based answer judging?

I initially implemented LLM-as-judge for answer quality assessment, but Llama 3.2 3B lacked discriminative capability - it consistently scored 3.5-4.0 regardless of actual quality variations. Rather than continue with unreliable metrics, I pivoted to focus on retrieval metrics (Recall@k, MRR, NDCG) which are:
- Objective and reproducible
- Industry-standard for RAG evaluation
- Show clear differences between strategies
- More aligned with production RAG practices

This decision is documented in [CHANGELOG.md](CHANGELOG.md) (Iteration 5 section).

### Does this work on Mac/Windows/Linux?

Yes. Docker Desktop works on all three platforms:
- **Linux:** Native Docker support + Make pre-installed
- **macOS:** Docker Desktop includes everything needed + Make pre-installed
- **Windows:** Docker Desktop with WSL2 backend. Make not included by default.

**Windows users:** See [Platform-Specific Notes](#platform-specific-notes) at the top for three setup options. The short version: either install Make via `choco install make`, or use the [Direct Docker Commands](#direct-docker-commands-advanced) instead of `make` commands.

### How much does the evaluation cost?

Zero dollars. Everything runs locally:
- Embedding model: Cached locally (~90 MB)
- LLM: Ollama with Llama 3.2 (~2 GB, free)
- Vector DB: ChromaDB (local storage)

The only "cost" is ~45 minutes of compute time to run the full evaluation.

### Can I use a different LLM model?

Yes. Edit `config/config.yaml` to specify your preferred model:

```yaml
generation:
  model: "llama3.1:8b"  # Larger, better quality
  # or "mistral:7b"
  # or "llama3.2:1b"  # Smaller, faster
```

Then download the model (may take several minutes depending on model size):
```bash
docker compose exec ollama ollama pull llama3.1:8b
```

### Can I use this on a different corpus?

Yes, the system is modular. You would need to:
1. Add your HTML/text documents to `data/corpus/`
2. Update HTML parser if format differs significantly from scikit-learn docs
3. Create new test set for evaluation (JSON file with question-answer pairs)
4. Run the full pipeline: `make preprocess` → `make index` → `make query`/`make eval`

---

## License

See [LICENSE](LICENSE) file for details.

---

## Related Links

- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Dataset Source:** https://scikit-learn.org/dev/versions.html (1.7.2 stable)
- **Part of ML Portfolio:** [ML-portfolio](../)

---

**Last Updated:** October 2025  
**Docker Support:** Linux, macOS, Windows  
**Total Setup Time:** ~10 minutes
