# TODO - FlowPrompt Roadmap

This document tracks remaining features and improvements planned for FlowPrompt.

## Completed in v0.3.0

### compare() Convenience API
- [x] One-call `compare()` function for prompt A/B testing
- [x] Async `acompare()` variant with parallel execution
- [x] `ComparisonResult` and `VariantResult` dataclasses
- [x] Top-level exports from `flowprompt`
- [x] Prompt Lab example (`examples/10_prompt_lab.py`)

### Structured Output Improvements
- [x] Native JSON schema mode via `litellm.supports_response_schema()`
- [x] Graceful fallback for models without native schema support

### Pricing
- [x] Dynamic pricing via `litellm.model_cost` instead of hardcoded table

### Documentation
- [x] README rewritten to lead with `compare()`
- [x] Quickstart updated with Compare Prompts section

## Completed in v0.2.0

### Automatic Optimization (DSPy-style)
- [x] Implement prompt optimization framework
- [x] Add automatic few-shot example selection
- [x] Support metric-driven prompt tuning
- [x] Integrate with Optuna for hyperparameter search
- [x] Add bootstrapping for self-improvement

### A/B Testing Framework
- [x] Create experiment configuration system
- [x] Implement traffic splitting logic
- [x] Add statistical significance testing
- [x] Build results dashboard/reporting
- [x] Support multi-armed bandit algorithms

### Multimodal Support
- [x] Image input support (vision models)
- [x] Audio input/output support
- [x] Video frame extraction and analysis
- [x] Document/PDF processing
- [x] Multi-image conversations

## Medium Priority

### Redis Cache Backend
- [ ] Implement Redis cache adapter
- [ ] Add connection pooling
- [ ] Support Redis Cluster
- [ ] Add cache invalidation patterns
- [ ] Implement distributed locking

### Langfuse Integration
- [ ] Add Langfuse tracing backend
- [ ] Implement session tracking
- [ ] Support user feedback collection
- [ ] Add prompt versioning sync
- [ ] Build cost attribution

### Environment Management
- [ ] Dev/staging/prod configuration
- [ ] Environment-specific model routing
- [ ] Feature flags for prompts
- [ ] Gradual rollout support
- [ ] Configuration validation

## Lower Priority

### Security Features

#### Prompt Injection Detection
- [ ] Implement injection pattern detection
- [ ] Add input sanitization utilities
- [ ] Create security scanning CLI command
- [ ] Support custom detection rules
- [ ] Add alerting/logging for attempts

#### PII Redaction
- [ ] Detect common PII patterns (SSN, email, phone)
- [ ] Implement automatic redaction
- [ ] Add reversible tokenization
- [ ] Support custom PII patterns
- [ ] Create audit logging

### Compliance & Audit

#### Audit Trails
- [ ] Log all prompt executions
- [ ] Track input/output pairs
- [ ] Implement retention policies
- [ ] Add export functionality
- [ ] Support compliance reporting (SOC2, HIPAA)

## Technical Debt

- [ ] Increase test coverage to 95%+
- [ ] Add integration tests for all providers
- [ ] Improve error messages and debugging
- [ ] Add performance benchmarks
- [ ] Create migration guides for updates

## Documentation

- [ ] Add more code examples
- [ ] Create video tutorials
- [ ] Write provider-specific guides
- [ ] Add troubleshooting section
- [ ] Create architecture diagrams

---

**Contributing**: We welcome contributions! Pick an item from this list and open a PR. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
