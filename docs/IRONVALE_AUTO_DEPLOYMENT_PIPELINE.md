# Ironvale Auto-Deployment Pipeline

## Separation Rule
- `research/` and `execution/` are separate concerns.
- Research reads logs and feature snapshots.
- Execution deploys only validated models.

## Live Data Path
`WebSocket -> in-memory state -> feature calc -> probability blend -> edge gate -> risk gate -> order router`

## Research Path
`historical data -> replay -> feature store -> train -> validate -> model registry`

## Model Registry Gate
Every candidate model must include:
- Version ID
- Train/validation windows
- Sharpe
- p-value
- Brier score
- Regime breakdown

Only `APPROVED` models are deployable.

## Retraining Cadence
- Weekly: rolling Sharpe + calibration refresh.
- Monthly: retrain flow layer, walk-forward test, Monte Carlo.
- Deployment flow: `research -> staging -> compare -> promote`.

## Guardrails
- Halt if drawdown exceeds configured threshold.
- Halt if exposure limits are breached.
- Cut Kelly when calibration drift is detected.
- Raise edge threshold during volatility spikes.

