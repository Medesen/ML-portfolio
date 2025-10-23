# ML Portfolio

A collection of machine learning projects I've built to demonstrate practical implementations, systematic evaluation, and production-oriented engineering practices. Each project includes complete documentation, reproducible Docker environments, and real-world datasets.

NB: This is a work in progress. Projects will be added gradually, but probably not very rapidly, as I'm writing this in my spare time.

## Current Projects

### 1. RAG Pipeline - Retrieval System with Systematic Evaluation

**Status:** Complete  
**Domain:** Technical documentation Q&A  
**Key Finding:** Fixed chunking achieved the highest retrieval performance (Recall@10: 0.51, MRR: 0.51)  
**Tech Stack:** ChromaDB, sentence-transformers, Ollama (Llama 3.2), Docker

A retrieval-augmented generation system I built to compare chunking strategies for technical documentation Q&A. I implemented and evaluated three chunking approaches (fixed, semantic, hierarchical) using standard IR metrics across 35 test questions. Everything runs locally in Docker with no API keys required.

**Highlights:**
- Systematic evaluation comparing three chunking strategies
- Complete Docker deployment with automated setup
- 35-question test set with IR metrics (Recall@k, MRR, NDCG)
- Runs entirely locally using Ollama for LLM generation
- ~10 minute setup time, fully reproducible

**[View Project →](rag_pipeline/)**

---

## What This Portfolio Demonstrates

### Engineering Practices

**Reproducibility:** I build all projects using Docker with automated setup scripts. Anyone can clone and run them without manual environment configuration or API dependencies.

**Documentation:** Each project includes comprehensive documentation covering architecture, design decisions, and iteration history. I write README files with quick-start guides, troubleshooting sections, and FAQs to make projects immediately usable.

**Evaluation:** I emphasize systematic evaluation using standard metrics. For example, the RAG pipeline uses industry-standard IR metrics (Recall@k, MRR, NDCG) rather than anecdotal assessment.

### Technical Skills

**ML/AI:** Retrieval-augmented generation, embedding models, vector databases, LLM integration, evaluation frameworks

**Software Engineering:** Modular architecture, dependency injection, configuration management, state tracking, logging, type hints, unit testing

**DevOps:** Docker & Docker Compose, automated deployment, multi-service orchestration, cross-platform compatibility (Linux/macOS/Windows)

**Data Engineering:** HTML parsing, chunking strategies, data preprocessing pipelines, batch processing

---

## Repository Structure

```
ML-portfolio/
├── README.md                    # Portfolio overview (this file)
├── rag_pipeline/                # Project 1: RAG system with evaluation
│   ├── README.md               # Complete documentation (633 lines)
│   ├── ARCHITECTURE.md         # System design patterns
│   ├── DESIGN.md               # Design decisions & trade-offs
│   ├── CHANGELOG.md            # Development iterations (1-5)
│   ├── src/                    # Source code (27 Python modules)
│   │   ├── preprocessing/      # HTML → JSON pipeline
│   │   ├── chunking/          # 3 chunking strategies
│   │   ├── retrieval/         # Embeddings & vector search
│   │   ├── generation/        # LLM integration
│   │   ├── evaluation/        # Metrics & analysis
│   │   └── utils/             # Config, logging
│   ├── tests/                  # 18 unit tests (21% coverage)
│   ├── data/
│   │   ├── corpus/            # Scikit-learn docs (416 files, tracked)
│   │   └── evaluation/        # Test set (35 Q&A pairs, tracked)
│   ├── config/
│   │   └── config.yaml        # YAML configuration
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Multi-service orchestration
│   ├── Makefile               # 20+ command shortcuts
│   └── setup.sh / setup.ps1   # Automated setup scripts
└── [future projects...]        # More projects coming soon
```

Each project is self-contained with its own:
- Detailed README with quick-start guide and examples
- Architecture and design documentation
- YAML-based configuration
- Unit test suite with pytest
- Docker environment with automated setup
- Complete dataset (where feasible and permitted by licensing)

---

## Future Projects

This portfolio is actively expanding. Planned additions include:

**LLM / AI / NLP:**
- **PEFT fine-tuning** - QLoRA domain adaptation, LoRA adapters, zero-shot comparison
- **LLM inference service** - FastAPI + vLLM, throughput/latency benchmarking, streaming
- **Prompt engineering cookbook** - Maintainable prompts, guardrails, regression tests
- **Classic NLP baselines** - TF-IDF, BiLSTM-CRF vs Transformers comparison

**Classic ML:**
- **End-to-end ML service** - Problem → model → API → CI/CD → monitoring
- **Tabular ML** - Feature engineering, experiment tracking, SHAP, model cards
- **Causal inference** - A/B testing, CUPED, uplift modeling
- **Time-series forecasting** - ARIMA/Prophet vs XGBoost vs deep learning
- **Recommender system** - Two-tower retrieval + ranking, offline metrics
- **Data pipeline orchestration** - Airflow/Prefect, data quality, versioning
- **Analytics & storytelling** - SQL, interactive viz, insights reports

Each new project will follow the same principles:
- Real-world datasets and problems
- Systematic evaluation and comparison
- Complete reproducibility
- Production-oriented engineering
- Comprehensive documentation

---

## Prerequisites

All projects in this portfolio use Docker for reproducibility and consistent environments across platforms:

- **Docker Desktop** ([Get Docker](https://docs.docker.com/get-docker/))
  - Includes Docker Compose (no separate install needed)
  - Works on Linux, macOS, and Windows
  - **Note:** Projects require Docker Compose V2 (`docker compose` command). If you have an older installation with only V1 (`docker-compose` with hyphen), either upgrade Docker or see project-specific troubleshooting sections for workarounds.
- **RAM:** 8GB minimum (12GB recommended)
- **Disk Space:** ~10GB free per project
- **Git** for cloning the repository

Each project's README contains specific prerequisites and platform-specific setup notes.

---

## Getting Started

### Quick Start

Each project has automated setup scripts that handle everything from building containers to downloading models and processing data. Here's the general workflow:

```bash
# Clone the repository
git clone https://github.com/Medesen/ML-portfolio.git
cd ML-portfolio

# Navigate to a specific project
cd rag_pipeline

# Run automated setup
./setup.sh        # Linux/macOS
.\setup.ps1       # Windows PowerShell

# The setup script will:
# - Build Docker containers
# - Download required models
# - Process datasets
# - Initialize the environment
```

### What Happens During Setup

Each project's setup script automates the entire environment configuration:
- **Container builds** (~2-3 minutes) - Creates isolated Docker environment
- **Model downloads** (varies by project) - Downloads required ML models
- **Data processing** (~1-2 minutes) - Prepares datasets for use
- **Verification** - Ensures everything is ready to run

After setup completes, you can immediately start using the project. All projects use Docker to ensure consistent environments across Linux, macOS, and Windows.

### Platform-Specific Notes

**Linux & macOS:** All commands work as shown. Docker and Make are typically pre-installed or easily available.

**Windows:** Projects include PowerShell setup scripts (`.ps1`) that work out of the box. Some projects use `make` commands for convenience, which requires installation (`choco install make`) or you can use direct Docker commands. See individual project READMEs for platform-specific details.

---

## Project Philosophy

### Why These Projects?

I focus on projects that demonstrate:

1. **Technical depth** - Not toy examples, but systems that address real challenges
2. **Engineering rigor** - Production-oriented code with testing, logging, and error handling
3. **Systematic evaluation** - Quantitative comparison using standard metrics
4. **Reproducibility** - Anyone can run them without API keys or complex setup
5. **Clear documentation** - Both technical details and high-level rationale

### Design Principles

**Local-first:** I design projects to run locally when feasible, eliminating API costs and ensuring reproducibility. Where cloud APIs would be used in production, I document the trade-offs explicitly.

**Data included:** I include datasets in the repository where licensing and size permit. This maximizes reproducibility and reduces setup friction.

**Honest trade-offs:** I explicitly cover decisions, limitations, and what would change for production deployment in each project's documentation. See each project's DESIGN.md for details.

**Iterative development:** I document the development process in CHANGELOG files, including pivots and lessons learned. This shows real-world problem-solving, not just polished final results.

---

## Technologies Used So Far

Based on the completed RAG pipeline project, I've demonstrated proficiency with:

**ML/AI Frameworks:**
- Sentence-transformers (embedding models)
- ChromaDB (vector database)
- Ollama (local LLM inference)
- PyTorch (underlying framework)

**Python Ecosystem:**
- Type hints and modern Python 3.12 features
- pytest for unit testing
- YAML-based configuration management
- Structured logging

**Development Tools:**
- Docker & Docker Compose for containerization
- Git version control
- Make for task automation
- Cross-platform compatibility (Linux/macOS/Windows)

**Evaluation & Metrics:**
- Information Retrieval metrics (Recall@k, MRR, NDCG)
- Systematic comparative evaluation
- Test set design and curation

As I add more projects, this section will expand to include additional technologies and frameworks.

---

## About This Portfolio

This portfolio showcases my approach to machine learning engineering: combining algorithmic understanding with software engineering best practices. I emphasize reproducibility, systematic evaluation, and clear documentation in every project.

I built these projects to demonstrate end-to-end capability - from problem definition and data processing through implementation, evaluation, and deployment. My focus is on production-oriented engineering rather than research prototypes.

---

## Contact & Links

- **GitHub:** [github.com/Medesen](https://github.com/Medesen)
- **Portfolio Repository:** [ML-portfolio](https://github.com/Medesen/ML-portfolio)

---

**Last Updated:** October 2025  
**Current Projects:** 1 complete, more coming soon  
**License:** See individual project directories for license details

