# What a correct result should look like (for checking the app)

Decisions (agreed):
1. Migrate the database to PostgreSQL
2. Deploy production on Kubernetes
3. No OAuth 2 upgrade this sprint (security review still open)

NOT decisions (must stay out of "Key decisions"): GraphQL (postponed), Redis caching (left for now).

Action items:
| Task | Owner | Deadline |
| Write the PostgreSQL migration script | Priya | Friday |
| Set up the CI/CD pipeline for Kubernetes | Arjun | unspecified (no date given) |
| Share the PyTorch benchmark results | Meera | next Tuesday |

NOT an action item: "somebody should look at Redis caching" (no owner, explicitly not scheduled).

Refiner check: raw transcript may contain things like "post gress", "cooper netties", "pie torch", "O off", "C I C D"; refined should say PostgreSQL, Kubernetes, PyTorch, OAuth, CI/CD. Numbers (92%, 400 ms) and negations ("not going to upgrade") must be unchanged.
