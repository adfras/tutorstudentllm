# Adaptive Scheduling (Domain-Agnostic)

- Track per-skill mastery in [0,1] with exponential decay (e.g., 7-day half-life).
- Schedule: remediate when incorrect → continue until mastered → review when due → advance when prerequisites satisfied.
- Misconceptions: maintain lightweight counters to prefer contrastive items.
- Supports closed-book notes accumulation as in-context memory for transfer tasks.
