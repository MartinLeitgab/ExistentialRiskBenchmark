# Test Suite Documentation

## Structure
```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_prompt_generator.py
│   ├── test_phase1_validation.py
│   ├── test_phase2_batch.py
│   └── test_result_analysis.py
└── integration/             # Integration tests (slower, E2E)
    ├── test_phase1_pipeline.py
    └── test_judge_integration.py
```

## Running Tests

### All tests
```bash
pytest tests/ -v
```

### Only unit tests
```bash
pytest tests/unit/ -v
```

### Only integration tests
```bash
pytest tests/integration/ -v
```

### Specific test file
```bash
pytest tests/unit/test_prompt_generator.py -v
```

### Specific test class
```bash
pytest tests/unit/test_prompt_generator.py::TestPromptGeneration -v
```

### Specific test
```bash
pytest tests/unit/test_prompt_generator.py::TestPromptGeneration::test_generate_prompt_basic -v
```

### With coverage
```bash
pytest tests/ --cov=pipeline_a_scenarios --cov-report=html
```

### By marker
```bash
pytest tests/ -v -m integration
pytest tests/ -v -m "not integration"  # Skip integration tests
```

### Parallel execution
```bash
pytest tests/ -v -n auto  # Requires pytest-xdist
```

## Quick Commands
```bash
# Fast unit tests only
make test-unit

# All tests with coverage
make test-cov

# Watch mode (requires pytest-watch)
ptw tests/unit/
```