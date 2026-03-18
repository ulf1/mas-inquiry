---
name: ai_assistant_guidelines
description: Interaction rules, behavior guidelines, and negative constraints for the AI assistant.
---

# AI Assistant Guidelines

This skill defines how the AI (Copilot/Antigravity) should interact, behave, and what it must avoid.

## 1. Interaction & Behavior

- **Conciseness:** Be brief. Omit explanations for standard code unless asked. *Exception: Provide comprehensive tests and needed documentation.*
- **Code First:** Provide the solution first, then the explanation if necessary.
- **Incremental Changes:** Keep diffs minimal and focused. Provide enough context to locate the change, but do not reprint the entire file unless necessary.
- **Path of Least Resistance:** Prefer framework-native or standard library solutions. Do not introduce new third-party dependencies without clear justification.
- **Project Consistency:** Follow existing project patterns strictly. Prefer modifying existing modules over creating new ones.
- **Refactoring:** Do not refactor unrelated code unless explicitly requested.
- **Backward Compatibility:** Preserve backward compatibility unless told otherwise.

## 2. What Copilot Must Avoid

- Introducing breaking changes without instruction.
- Overengineering simple solutions.
- Adding speculative abstractions.
- Rewriting stable working code unnecessarily.
- Ignoring established architectural patterns.
