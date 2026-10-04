# Architecture

```text
                         PUBLIC USER
                             │
                    upload / camera / PDF
                             │
                    ┌────────┴────────┐
                    │                 │
             ROCK VISION       DOCUMENT AI
                    │                 │
             QC/normalize       validate/route
                    │                 │
            semantic mask       text layer / OCR
                    │                 │
             watershed          entity extraction
                    │                 │
        object measurements      page provenance
                    │                 │
                    └────────┬────────┘
                             │
                  TRACEABLE RESULT MODEL
                             │
          source → page → image → object → class
                             │
                   CSV / JSON / annotated PNG
```

Research and training remain in the notebook; deployment imports inference-only modules. Advanced models are optional and must never break the P0 path.
