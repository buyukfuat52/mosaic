# YOLO Mosaic AI Prompt Pack

This package contains English-language prompts and project guidance for instructing an AI coding tool to build a portfolio-ready YOLO Mosaic Annotation Toolkit.

## Files

- `SYSTEM_PROMPT.md` — primary system or developer prompt
- `PROJECT_SPEC.md` — concise technical specification
- `ACCEPTANCE_CRITERIA.md` — completion checklist
- `RUN_PROMPT.md` — initial user prompt for the coding tool
- `AGENTS.md` — optional repository-level instructions
- `PORTFOLIO_GUIDE.md` — resume, GitHub, LinkedIn, and interview guidance

## Recommended workflow

### Tools with a system prompt field

1. Place the contents of `SYSTEM_PROMPT.md` in the system or developer instruction field.
2. Copy the remaining files into the repository root.
3. Send the contents of `RUN_PROMPT.md` as the first user message.

### Tools without a system prompt field

Provide the files in this order:

1. `SYSTEM_PROMPT.md`
2. `PROJECT_SPEC.md`
3. `ACCEPTANCE_CRITERIA.md`
4. `RUN_PROMPT.md`

## Is a multi-agent setup required?

No.

A single capable coding model is sufficient if it can:

- inspect repository files
- create and edit files
- run terminal commands
- execute tests and quality checks
- inspect failures
- iterate until the project passes

A multi-agent setup may add coordination overhead without improving the core geometry work.

`AGENTS.md` is still useful for repository-based coding tools because it provides persistent development rules.

## Portfolio emphasis

The prompt requires:

- modular architecture
- numerical geometry tests
- GitHub Actions CI
- Docker
- pre-commit
- benchmark support
- synthetic demos
- real example outputs
- architecture documentation
- resume bullet templates
- technical interview talking points

The implementation must not invent benchmark, coverage, test, or CI results.
