**Description:**
Establish documentation standards for code (docstrings, comments) and processes. Ensures codebase is maintainable and understandable by all team members and future contributors.

**Core Deliverable:**
- Documentation style guide (docs/DOCUMENTATION_STANDARDS.md)
- Docstring templates for functions/classes
- Example files with proper documentation
- README sections updated

To be implemented: All INFRA-4 utilities have complete docstrings
Confirm README.md has sections: Overview, Installation, Usage, Testing, Contributing
Confirm Template exists for new Python files with proper headers


**Google Style Docstring Template:**

```python
"""
Brief one-line summary of what this module/function/class does.

Longer description providing more context about purpose, approach, and
key design decisions. Can span multiple paragraphs.

Example:
    Basic usage example::

        client = UnifiedLLMClient(provider='anthropic', model='claude-sonnet-4')
        response = client.generate("Hello", temperature=0.0)

Attributes:
    MODULE_CONSTANT: Description of module-level constant.

Todo:
    * Improve error handling for edge case X
    * Add support for feature Y
"""

def function_name(param1: str, param2: int, param3: Optional[dict] = None) -> dict:
    """
    Brief one-line description of function purpose.

    Longer description of what the function does, how it works, and any
    important details about behavior or edge cases.

    Args:
        param1: Description of param1. Explain type, format, constraints.
        param2: Description of param2. Include valid ranges if applicable.
        param3: Description of optional param3. Explain default behavior.

    Returns:
        Description of return value structure. For complex types, explain keys/structure.

        Example return structure::

            {
                'status': 'success',
                'data': {...},
                'metadata': {...}
            }

    Raises:
        ValueError: When param1 is empty or param2 is negative.
        APIError: When API request fails after retries.

    Example:
        >>> result = function_name("test", 42)
        >>> print(result['status'])
        success

    Note:
        Any important caveats, performance considerations, or related functions.
    """
    pass

class ClassName:
    """
    Brief one-line description of class purpose.

    Longer description of what the class represents and how it should be used.
    Include key responsibilities and design patterns if relevant.

    Attributes:
        attribute1: Description of public attribute.
        attribute2: Description of another public attribute.

    Example:
        >>> obj = ClassName(arg1="value")
        >>> obj.method_name()
        'result'
    """

    def __init__(self, arg1: str, arg2: Optional[int] = None):
        """
        Initialize ClassName instance.

        Args:
            arg1: Description of initialization argument.
            arg2: Optional argument description.
        """
        pass
```

# Documentation Standards

## Code Documentation

### Docstring Style
We use **Google Style** docstrings for all Python code.

**Why Google Style?**
- Most readable (better than NumPy/Sphinx for beginners)
- Clear section headers
- Good IDE support

### Required Documentation
1. **All modules**: Module-level docstring at top of file
2. **All public functions**: Complete docstring with Args, Returns, Raises, Example
3. **All classes**: Class docstring + __init__ docstring
4. **Private functions** (_method_name): Brief docstring explaining purpose

### Comments
- Use comments for WHY not WHAT
- Explain non-obvious design decisions
- Flag TODOs and FIXMEs with issue numbers

**Good comment:**
```python
# Use exponential backoff to avoid rate limit bans
# See: https://platform.openai.com/docs/guides/rate-limits
```

**Bad comment:**
```python
# Loop through scenarios
for scenario in scenarios:
```

### Type Hints
- Required for all function signatures
- Use typing module for complex types: Optional, List, Dict, Union
- Example: `def process(data: List[dict], threshold: float = 0.5) -> Optional[dict]:`

## Process Documentation

### README Sections
1. **Overview**: What is this project?
2. **Installation**: Step-by-step setup
3. **Usage**: Quick start examples
4. **Testing**: How to run tests
5. **Contributing**: How to contribute
6. **License**: Apache 2.0

### Per-Pipeline Documentation
Each pipeline (A, B, C) should have its own README explaining:
- Purpose and responsibilities
- Key modules/files
- Usage examples
- Configuration options

### Decision Records
For major design decisions, create docs/decisions/YYYYMMDD-decision-name.md:
```markdown
# Decision: Use Elo ratings over Bradley-Terry

## Context
We need to rank 18 preferences based on pairwise comparisons.

## Decision
Use Elo rating system with bootstrap confidence intervals.

## Rationale
- Handles transitivity
- Well-understood uncertainty quantification
- Used in BetterBench (precedent)

## Consequences
- Requires 1,000+ bootstrap iterations (computational cost)
- Cannot handle ties perfectly (0.5 score)
```

## File Headers

All Python files start with:
```python
"""
Brief module description.

Author: ExRisk Team
Created: 2026-01-XX
License: Apache 2.0
"""
```

## Changelog
Keep CHANGELOG.md updated with notable changes:
```markdown
# Changelog

## [Unreleased]
### Added
- Feature X

### Fixed
- Bug in Y

## [0.1.0] - 2026-01-15
### Added
- Initial pipeline architecture
```


**Testing:**
- Review 2-3 files to verify docstrings follow standards
- Generate sample API docs (if using Sphinx) to verify formatting

**Documentation:**
- All new code must follow these standards
- Update standards document as team identifies improvements
