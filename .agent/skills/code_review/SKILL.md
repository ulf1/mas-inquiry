---
name: code_review
description: Instructions and checklists for performing code reviews across the repository.
---

# Code Review Standards

This skill guides Copilot code review across all files in this repository. Use these standards to ensure security, performance, and code quality.

## 1. Security Critical Issues

- **Secrets:** Check for hardcoded secrets, API keys, or credentials.
- **Injection:** Look for SQL injection and XSS vulnerabilities.
- **Validation:** Verify proper input validation and sanitization.
- **Auth:** Review authentication and authorization logic.

## 2. Performance Red Flags

- **N+1 Queries:** Identify N+1 database query problems.
- **Loops:** Spot inefficient loops and algorithmic issues.
- **Resources:** Check for memory leaks and resource cleanup.
- **Caching:** Review caching opportunities for expensive operations.

## 3. Code Quality Essentials

- **Function Size:** Verify functions do not exceed 50 lines.
- **Naming:** Check that naming conventions matching the project style are followed.
- **Error Handling:** Ensure error handling covers edge cases and avoids silent failures.
- **Dead Code:** Flag any dead code or unused imports for removal.

## 4. Review Style

- **Actionable:** Be specific and actionable in feedback.
- **Why:** Explain the "why" behind recommendations.
- **Positive:** Acknowledge good patterns when you see them.
- **Questions:** Ask clarifying questions when code intent is unclear.

## 5. Testing Standards

- **Requirement:** New features require unit tests.
- **Coverage:** Tests should cover edge cases and error conditions.
- **Naming:** Test names should clearly describe what they test.

**Priority:** Always prioritize security vulnerabilities and performance issues that could impact users.

## 6. Review Checklist
- [ ] Code follows style guides
- [ ] Tests are present and passing
- [ ] No obvious security vulnerabilities
- [ ] No hardcoded secrets or credentials
- [ ] Error handling is implemented
- [ ] Code is documented where necessary
- [ ] No unnecessary dependencies added
