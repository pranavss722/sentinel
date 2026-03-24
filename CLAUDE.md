# ML Serving Platform — Dev Rules

## Workflow
Brainstorm → Plan → TDD → Standards

## Rules
- TDD: Write a failing test before any implementation file. No exceptions.
- YAGNI/KISS: No abstractions that aren't tested and required right now.
- CANARY: Traffic-split only (Bernoulli draw per request). Never feature-flagged.
- ROLLBACK: Automatic on SLO breach. SLO = p99 latency >200ms OR error rate >1%.
            Enforced in CanaryController only — never in middleware.
- COUPLING: Router, ModelRegistry, CanaryController, DriftMonitor inject
            dependencies — never import each other directly.
