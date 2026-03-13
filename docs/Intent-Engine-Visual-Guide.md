# INTENT ENGINE: Visual Architecture Guide & Quick Reference

**Version:** 2.0
**Format:** System Design Companion (PDF-Ready)
**Date:** February 17, 2026

---

## QUICK START: Intent Engine at a Glance

### What Problem Does It Solve?

| Problem | Google/Meta Approach | Intent Engine Approach |
|---------|-----|-----|
| How to rank search results? | User history + behavioral profile | Current intent + constraints |
| How to match ads? | Track user across web | Match on declared intent only |
| How to avoid discrimination? | Hope ML fairness works | Explicit constraints, no sensitive attributes |
| How to ensure privacy? | Bury tracking in 40-page ToS | Non-tracking by design |
| Can users control data? | Settings menu is hidden | Intent signals visible & editable |

### Core Insight

**Instead of "Who is this user?" → "What does this user want right now?"**

```
Google:  [User ID] → [Full profile] → [Personalized results] + [Targeted ads]
                    (Years of data)  (High tracking risk)

Intent:  [Query/Action] → [Intent schema] → [Matched results/ads]
                        (No history)     (Session-scoped)
```

---

## SYSTEM LAYERS (Visual)

### Layer 1: Capture (Sources of Intent)

```
┌─────────────────────────────────────────────────────────────┐
│                      INTENT SOURCES                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Search Query]    [Docs Edit]    [Mail Compose]            │
│  "How to setup      Opens "Q1      Drafts email to          │
│   E2E email?"       Budget"        contractors              │
│  ↓                  ↓              ↓                        │
│  Text              Document        Email                    │
│  Parsing           State           Metadata                 │
│                                                             │
│  [Calendar]        [Diary]        [Meet]                    │
│  Creates event     Writes:        Joins video call         │
│  "Weekly 1:1"      "Career        with stakeholders        │
│  ↓                  confusion"     ↓                        │
│  Structured        Free form       Metadata +              │
│  Event             text            Participants            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Layer 2: Extract & Infer (Intelligence)

```
┌─────────────────────────────────────────────────────────────┐
│               INTENT EXTRACTION ENGINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT: "How to setup E2E encrypted email on Android,      │
│          no big tech"                                       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ PHASE 1: PARSING (Regex + NLP + Rules)              │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ • Extract constraints: Android, E2E, no Google/MS   │  │
│  │ • Extract goal: LEARN                               │  │
│  │ • Extract negatives: "no big tech"                  │  │
│  │ • Skill level: intermediate (implied by "setup")    │  │
│  └──────────────────────────────────────────────────────┘  │
│                     ↓                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ PHASE 2: INFERENCE (Context + Patterns)             │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ • Use cases: Learning, troubleshooting              │  │
│  │ • Temporal: Today, recent, one-off                 │  │
│  │ • Ethics: Privacy-first, open-source preference    │  │
│  │ • Result type: Tutorial                             │  │
│  │ • Complexity: Moderate                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                     ↓                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ PHASE 3: NORMALIZATION (Map to Schema)              │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ OUTPUT: UniversalIntent JSON object                 │  │
│  │ ├─ declared: {goal, constraints, skill_level}      │  │
│  │ ├─ inferred: {use_cases, temporal, ethics}         │  │
│  │ └─ expires_at: 2026-01-19T20:34:56Z (session end)  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Layer 3: Rank & Match (Output)

```
┌─────────────────────────────────────────────────────────────┐
│          RANKING & MATCHING USING INTENT                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SEARCH RANKING                                             │
│  ┌────────────────────────────────────┐                    │
│  │ Hard Filters (Constraints)         │                    │
│  │ ✗ Exclude: Google, Microsoft       │                    │
│  │ ✓ Include: Android, E2E Encrypted  │                    │
│  │                                    │                    │
│  │ Soft Scoring (Intent Alignment)    │                    │
│  │ • Goal match: Tutorial (+0.25)     │                    │
│  │ • Use case: Learning (+0.15)       │                    │
│  │ • Skill level: Intermediate (+0.20)│                    │
│  │ • Quality: Recency, authority      │                    │
│  │ • Ethical: Privacy-respecting site │                    │
│  │                                    │                    │
│  │ FINAL RANKING:                     │                    │
│  │ 1. ProtonMail guide (0.92) ←────────────│ Highest match
│  │ 2. Tutanota tutorial (0.88)        │    │
│  │ 3. Mailbox.org setup (0.85)        │    │
│  └────────────────────────────────────┘    │
│                                             │
│  AD MATCHING                                │
│  ┌────────────────────────────────────┐   │
│  │ No tracking, no behavioral profile │   │
│  │                                    │   │
│  │ Intent: LEARN about E2E email     │   │
│  │ Constraints: Open-source, privacy │   │
│  │                                    │   │
│  │ Matched ads:                       │   │
│  │ ✓ Tutanota (privacy email)        │   │
│  │ ✓ Proton VPN (privacy company)    │   │
│  │ ✗ Gmail Ad (violates constraints) │   │
│  └────────────────────────────────────┘   │
│                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## UNIVERSAL INTENT SCHEMA (Visual)

```
┌──────────────────────────────────────────────────────────────────┐
│                    UNIVERSAL INTENT OBJECT                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  intentId:         "sess_abc123_1" (random, session-scoped)     │
│  expiresAt:        "2026-01-19T20:34:56Z" (session end)         │
│  product:          "search" | "docs" | "mail" | ...             │
│  sessionId:        "sess_abc123" (not linked across sessions)   │
│  timestamp:        "2026-01-19T12:34:56Z"                       │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  DECLARED INTENT (What user explicitly asked for)                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  query:           "How to setup end-to-end encrypted email      │
│                   on Android, no big tech solutions"            │
│                                                                  │
│  goal:            LEARN (enumerated: FIND_INFO, COMPARISON,     │
│                          TROUBLESHOOTING, PURCHASE, ...)       │
│                                                                  │
│  constraints:     [                                              │
│                     {type: "inclusion", dimension: "platform",   │
│                      value: "Android", hardFilter: true},        │
│                     {type: "inclusion", dimension: "feature",    │
│                      value: "E2E_encryption", hardFilter: true}, │
│                     {type: "exclusion", dimension: "provider",   │
│                      value: ["Google", "Microsoft"],            │
│                      hardFilter: true}                           │
│                   ]                                              │
│                                                                  │
│  negativePreferences: ["no big tech", "no proprietary"]         │
│  urgency:         "soon" (immediate | soon | flexible)          │
│  skillLevel:      "intermediate"                                 │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  INFERRED INTENT (What we derived from context)                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  useCases:        ["learning", "troubleshooting"]               │
│  resultType:      "tutorial"                                    │
│  complexity:      "moderate"                                    │
│                                                                  │
│  temporalIntent: {                                              │
│    horizon: "today",        (immediate | today | week | ...)   │
│    recency: "recent",       (breaking | recent | evergreen)    │
│    frequency: "oneoff"      (oneoff | recurring | exploratory) │
│  }                                                               │
│                                                                  │
│  ethicalSignals: [                                              │
│    {dimension: "privacy", preference: "privacy-first"},        │
│    {dimension: "openness", preference: "open-source_preferred"}│
│  ]                                                               │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  SESSION FEEDBACK (Captured during session, deleted at end)      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  clicked:        ["https://protonmail.com/...", ...]           │
│  dwellTime:      45 (seconds)                                  │
│  bounced:        false                                          │
│  reformulated:   true (user refined query)                     │
│                                                                  │
│  ⚠️ This data is NOT PERSISTED after session ends              │
│     (No user history, no behavioral profiling)                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## DATA LIFECYCLE (Privacy Guarantees)

```
┌─────────────────────────────────────────────────────────────┐
│                  INTENT DATA LIFECYCLE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  T0: User enters query (device local)                      │
│      ↓                                                     │
│      Extract intent (rules-based, no ML profiling)         │
│      ↓                                                     │
│      Store in-memory: UniversalIntent object               │
│      ├─ ttl: 8 hours                                       │
│      ├─ sessionId: Random, not linked to user ID           │
│      └─ expiresAt: [NOW + 8 hours]                         │
│      ↓                                                     │
│  T1: User browses results (session ongoing)                │
│      ├─ Clicks registered locally                          │
│      ├─ Dwell time measured                                │
│      ├─ sessionFeedback object updated (in-memory)         │
│      └─ NO data sent to server yet                         │
│      ↓                                                     │
│  T2: Session ends (browser close, logout, 8 hours pass)    │
│      ├─ Intent object: AUTO-DELETED                        │
│      ├─ Session feedback: AUTO-DELETED                     │
│      └─ sessionId: PURGED                                  │
│      ↓                                                     │
│  Server-side (if logging)                                  │
│      ├─ Aggregated ranking log: 7-day retention            │
│      ├─ Aggregated ad metrics: 30-day retention            │
│      ├─ Both: ENCRYPTED at rest                            │
│      └─ Auto-purge after TTL                               │
│                                                             │
│  ✓ Privacy Guarantee: No user identified across sessions   │
│  ✓ No behavioral profile created                           │
│  ✓ No cross-product user linking                           │
│  ✓ No third-party data integration                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## CONSTRAINT SATISFACTION (Hard Filters)

```
┌─────────────────────────────────────────────────────────────┐
│         HARD FILTERS: Constraints in Action                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User declares:                                            │
│  ✓ INCLUDE: Android platform                              │
│  ✓ INCLUDE: E2E encryption                                │
│  ✗ EXCLUDE: Google, Microsoft                             │
│  ✗ EXCLUDE: Proprietary license                           │
│  Budget: Under 1000 rupees                                │
│                                                             │
│  Result 1: "ProtonMail for Android"                        │
│  ├─ Android? ✓ Yes                                        │
│  ├─ E2E Encrypted? ✓ Yes                                  │
│  ├─ Google/Microsoft? ✗ No (independent company)          │
│  ├─ Open Source? ✓ Partially (client open)               │
│  ├─ Price? ✓ Free tier available                          │
│  └─ PASSES FILTER ✓ Rank this result                      │
│                                                             │
│  Result 2: "Gmail with encryption"                         │
│  ├─ Android? ✓ Yes                                        │
│  ├─ E2E Encrypted? ⚠️ Optional (not default)              │
│  ├─ Google/Microsoft? ✗ YES (Google) ← FAILS              │
│  └─ FAILS FILTER ✗ Don't rank (exclude immediately)       │
│                                                             │
│  Result 3: "Tutanota"                                      │
│  ├─ Android? ✓ Yes                                        │
│  ├─ E2E Encrypted? ✓ Yes                                  │
│  ├─ Google/Microsoft? ✗ No (independent)                  │
│  ├─ Open Source? ✓ Yes (client & server)                 │
│  ├─ Price? ✓ Free tier available                          │
│  └─ PASSES FILTER ✓ Rank this result                      │
│                                                             │
│  ═══════════════════════════════════════════════════════════
│  Ranking by intent alignment (after hard filter):          │
│  1. Tutanota (0.92) – Full match on all constraints       │
│  2. ProtonMail (0.88) – Missing only open-source server   │
│  3. [Gmail excluded due to constraint violation]           │
│                                                             │
│  KEY: Hard filters prevent irrelevant results BEFORE       │
│       scoring. No user sees results that violate their      │
│       explicit preferences.                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## INTENT-BASED AD MATCHING (No Tracking)

```
┌──────────────────────────────────────────────────────────────┐
│           HOW ADS WORK IN INTENT ENGINE                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  GOOGLE APPROACH (Tracking-Based):                           │
│  ┌──────────────────────────────────────┐                   │
│  │ 1. Track user across web             │                   │
│  │ 2. Build behavioral profile          │                   │
│  │ 3. Predict interests (ML)            │                   │
│  │ 4. Sell to advertisers for targeting │                   │
│  │ 5. Match ads based on profile        │                   │
│  │                                      │                   │
│  │ Privacy Risk: Fingerprinting,        │                   │
│  │ cross-site tracking, data brokers    │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
│  INTENT ENGINE APPROACH (Declared Intent):                   │
│  ┌──────────────────────────────────────┐                   │
│  │ 1. Extract current intent (session)  │                   │
│  │ 2. Respect user constraints          │                   │
│  │ 3. Match ads to intent (no history)  │                   │
│  │ 4. Delete intent at session end      │                   │
│  │ 5. No user profile built             │                   │
│  │                                      │                   │
│  │ Privacy: No tracking, no profiling,  │                   │
│  │ session-scoped only                  │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  MATCHING ALGORITHM (Simplified)                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Intent: User searches for "budget-friendly E2E mail"       │
│                                                              │
│  Ad Filter 1: User Constraints                              │
│  ├─ User excluded: [Gmail, Outlook, Apple]                  │
│  └─ Filter ads from these providers                         │
│                                                              │
│  Ad Filter 2: Advertiser Constraints (Fairness)             │
│  ├─ Allowed: Geo, device type, language, declared intent   │
│  ├─ BANNED: Age, gender, income, health, behavior          │
│  └─ Gmail ad targeting "age 25-34" → REJECTED              │
│                                                              │
│  Ad Scoring: Relevance (no tracking)                        │
│  Ad 1: Tutanota email                                       │
│  ├─ Intent goal match (email service): +0.40               │
│  ├─ Use case match (budgeting): +0.15                      │
│  ├─ Ethical alignment (privacy-first): +0.20              │
│  ├─ Keyword match: +0.10                                   │
│  └─ TOTAL: 0.85 (high relevance)                           │
│                                                              │
│  Ad 2: VPN service                                          │
│  ├─ Intent goal match (email service): +0.05               │
│  ├─ Ethical alignment: +0.20                               │
│  └─ TOTAL: 0.25 (low relevance)                            │
│                                                              │
│  Final: Show Tutanota ad (highest relevance, no tracking)  │
│                                                              │
│  ✓ Better for user: Relevant ads without surveillance      │
│  ✓ Better for publisher: Higher CTR (users engage)         │
│  ✓ Better for advertiser: Qualified customers              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## SERVICE RECOMMENDATION (Cross-Product Intent)

```
┌──────────────────────────────────────────────────────────────┐
│        ROUTING USER TO RIGHT SERVICE                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Scenario: User opens workspace with ambiguous intent       │
│  Input: "Q1 budget, need to collaborate with team"          │
│                                                              │
│  Intent Analysis:                                            │
│  ├─ Goal: COLLABORATION                                     │
│  ├─ Secondary: ANALYZE                                      │
│  ├─ Data type: SPREADSHEET (inferred)                       │
│  └─ Urgency: SOON                                           │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  Service Matching (Scoring)                                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Service        │ Score │ Reason                              │
│  ────────────────┼───────┼──────────────────────────────────   │
│  Docs           │ 0.95  │ Best for collaboration + spreadsheet│
│  ├─ Collab goal │ +0.60 │ Primary feature                     │
│  ├─ Analyze     │ +0.20 │ Secondary use case                  │
│  ├─ Data type   │ +0.15 │ Spreadsheet editing                 │
│                                                              │
│  Sheets         │ 0.88  │ Good for spreadsheet work          │
│  ├─ Analyze     │ +0.50 │ Primary feature                     │
│  ├─ Collab goal │ +0.30 │ But weaker collab UX                │
│  ├─ Data type   │ +0.08 │ Native spreadsheet                  │
│                                                              │
│  Mail           │ 0.42  │ Lower (collaboration via email)    │
│  ├─ Collab      │ +0.20 │ Possible but inefficient            │
│  ├─ Analyze     │ +0.10 │ Not a primary use case              │
│                                                              │
│  Search         │ 0.15  │ Not relevant to this intent         │
│  ├─ Find info   │ +0.15 │ Not the goal                        │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  RECOMMENDATION                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Primary:   Docs (0.95)   ← Open this first                 │
│  Secondary: Sheets (0.88) ← Offer as alternative            │
│  Lower:     Mail (0.42)   ← Available if user wants          │
│                                                              │
│  ✓ Route user to right tool without tracking their behavior │
│  ✓ No need for profile ("users like you prefer X")          │
│  ✓ Purely intent-driven routing                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## THREAT MODEL SUMMARY

```
┌──────────────────────────────────────────────────────────────┐
│              PRIVACY THREATS & MITIGATIONS                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Threat 1: Intent Inference Side Channels                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Attack: Adversary reconstructs user behavior from    │   │
│  │         repeated queries across sessions             │   │
│  │ Example: Weekly "IVF budgeting" queries leak         │   │
│  │          health intent                               │   │
│  │                                                      │   │
│  │ Mitigation:                                          │   │
│  │ ✓ Intent TTL = 8 hours (session-scoped)              │   │
│  │ ✓ No query history stored                            │   │
│  │ ✓ SessionId randomized (not linked to user ID)       │   │
│  │ ✓ No cross-session intent correlation                │   │
│  │ ✓ Each session is fresh intent context               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Threat 2: Ad Matching Privacy Leakage                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Attack: Advertiser deduces user intent from ad       │   │
│  │         metrics                                       │   │
│  │ Example: Advertiser A gets 100 impressions for       │   │
│  │          "antidepressant" ads → infers user health   │   │
│  │                                                      │   │
│  │ Mitigation:                                          │   │
│  │ ✓ No persistent audience segments                    │   │
│  │ ✓ Advertiser sees: aggregate metrics only            │   │
│  │ ✓ No per-user ad targeting info shared               │   │
│  │ ✓ Differential privacy on conversions                │   │
│  │ ✓ Noise added to ad metrics for privacy              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Threat 3: Document Context Leakage                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Attack: System reads doc content to profile users    │   │
│  │ Example: "Cancer Research Proposal" in Docs name     │   │
│  │          → infer user health status                  │   │
│  │                                                      │   │
│  │ Mitigation:                                          │   │
│  │ ✓ Read document titles only (not content)            │   │
│  │ ✓ First 1000 chars of content for context            │   │
│  │ ✓ Content not persisted                              │   │
│  │ ✓ Don't profile individual documents                 │   │
│  │ ✓ User can opt-out of intent inference from docs     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Threat 4: Cross-Product Intent Linking                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Attack: Combine intents across services to profile   │   │
│  │         users                                         │   │
│  │ Example: Search "apartments" + Mail from realtor     │   │
│  │          + Docs "Housing Plans"                      │   │
│  │          → sell real estate ads                       │   │
│  │                                                      │   │
│  │ Mitigation:                                          │   │
│  │ ✓ Intent objects are product-scoped                  │   │
│  │ ✓ No cross-product intent aggregation                │   │
│  │ ✓ Each product has independent session               │   │
│  │ ✓ Users can't be profiled across services            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Threat 5: Temporal Inference                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Attack: Correlate timing of intents across sessions  │   │
│  │ Example: Doc "Resignation Letter" created 3pm        │   │
│  │          → infer job transition → advertise          │   │
│  │          executive recruiter services                │   │
│  │                                                      │   │
│  │ Mitigation:                                          │   │
│  │ ✓ SessionId expires after 8 hours                    │   │
│  │ ✓ No persistent session links                        │   │
│  │ ✓ Each new session is fresh context                  │   │
│  │ ✓ No temporal correlation across sessions            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## FAIRNESS CONSTRAINTS (No Discrimination)

```
┌──────────────────────────────────────────────────────────────┐
│        HOW INTENT ENGINE PREVENTS DISCRIMINATION             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Google/Meta Problem:                                        │
│  ┌──────────────────────────────────────┐                   │
│  │ Behavioral profiling → Proxy           │                   │
│  │ discrimination                         │                   │
│  │                                        │                   │
│  │ Example 1:                             │                   │
│  │ Users from low-income ZIP codes        │                   │
│  │ → Shown lower-wage job ads             │                   │
│  │ (Discriminatory even if no explicit    │                   │
│  │  age/income targeting)                 │                   │
│  │                                        │                   │
│  │ Example 2:                             │                   │
│  │ Female users → Shown fewer             │                   │
│  │ high-wage tech jobs                    │                   │
│  │ (Gender-based algorithmic bias)        │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
│  Intent Engine Solution:                                     │
│  ┌──────────────────────────────────────┐                   │
│  │ 1. Explicit constraints only          │                   │
│  │    (no proxy inference)                │                   │
│  │                                        │                   │
│  │ 2. Forbidden targeting dimensions      │                   │
│  │    - Age, gender, race, religion       │                   │
│  │    - Income, credit score              │                   │
│  │    - Health conditions                 │                   │
│  │    - Behavioral segments               │                   │
│  │                                        │                   │
│  │ 3. Allowed targeting dimensions        │                   │
│  │    - Geographic region (India, EU)     │                   │
│  │    - Device type (mobile, desktop)     │                   │
│  │    - Language (English, Hindi)         │                   │
│  │    - Declared intent (search query)    │                   │
│  │                                        │                   │
│  │ 4. Constraint auditing                 │                   │
│  │    - All advertiser constraints        │                   │
│  │      checked against forbidden list    │                   │
│  │    - Discriminatory ads rejected       │                   │
│  │    - Fairness logged & monitored       │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  EXAMPLE: Ad Matching with Fairness Check                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Ad: "McKinsey Consulting - Senior Analyst Role"             │
│                                                              │
│  Advertiser Constraints:                                     │
│  ├─ Geographic: India ✓ (ALLOWED)                            │
│  ├─ Device: Mobile ✓ (ALLOWED)                               │
│  ├─ Declared intent: "job search" ✓ (ALLOWED)                │
│  └─ Gender targeting: Female only ✗ (BANNED!)               │
│                                                              │
│  System Action:                                              │
│  ├─ Detect gender constraint                                │
│  ├─ Reject ad (fairness violation)                          │
│  ├─ Log violation for audit                                 │
│  └─ Notify advertiser: "Discriminatory constraint removed"  │
│                                                              │
│  Result: Ad shown to all qualified users (no gender bias)   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## COMPLIANCE CHECKLIST

```
┌──────────────────────────────────────────────────────────────┐
│         REGULATORY COMPLIANCE STATUS                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  GDPR (EU Data Protection)                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Requirement              │ Intent Engine             │   │
│  │ ────────────────────────┼───────────────────────────  │   │
│  │ Right to deletion        │ ✓ Auto-delete at session end │   │
│  │ Consent for tracking     │ ✓ No tracking needed        │   │
│  │ Data portability         │ ✓ Export intent as JSON     │   │
│  │ Purpose limitation       │ ✓ Intent = matching only    │   │
│  │ Data minimization        │ ✓ No persistent profiles    │   │
│  │ Accountability           │ ✓ Fairness audits logged    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  CCPA (California Consumer Privacy)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Requirement              │ Intent Engine             │   │
│  │ ────────────────────────┼───────────────────────────  │   │
│  │ Know what data is       │ ✓ Intent schema documented  │   │
│  │ collected                                             │   │
│  │ Delete personal info    │ ✓ Auto-delete, user request │   │
│  │ Opt-out of "sale"       │ ✓ Disable ad matching       │   │
│  │ Non-discrimination      │ ✓ Fairness constraints      │   │
│  │ No price discrimination │ ✓ Explicit constraint check │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  India DPDP Act (Digital Personal Data Protection)           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Requirement              │ Intent Engine             │   │
│  │ ────────────────────────┼───────────────────────────  │   │
│  │ Processing transparency │ ✓ Extraction rules public   │   │
│  │ Consent & withdrawal    │ ✓ User control of intent    │   │
│  │ Purpose limitation      │ ✓ Intent expires session    │   │
│  │ Data minimization       │ ✓ No behavioral tracking    │   │
│  │ Anonymization           │ ✓ Intent not linked to ID   │   │
│  │ Right to grievance      │ ✓ Fairness audit complaints │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Overall Privacy Score: ✓ COMPLIANT (by design)             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## COMPARISON: Intent Engine vs. Google vs. Meta

```
┌──────────────────────────────────────────────────────────────┐
│             SYSTEM COMPARISON MATRIX                         │
├──────────────────┬────────────────┬──────────────┬───────────┤
│ Dimension        │ Google         │ Meta         │ Intent    │
│                  │                │              │ Engine    │
├──────────────────┼────────────────┼──────────────┼───────────┤
│ Data Collection  │ Cross-site     │ Cross-site   │ Declared  │
│                  │ tracking       │ tracking     │ intent    │
│                  │ (Cookies, GA)  │ (Pixels)     │ (No track)│
│                  │                │              │           │
│ Signal Lifetime  │ 2-3 years      │ 90 days      │ < 8 hrs   │
│                  │ (Persistent)   │ (Long-term)  │ (Session) │
│                  │                │              │           │
│ User Control     │ ⚠️ Limited     │ ⚠️ Limited   │ ✓ Native  │
│                  │ (Settings      │ (Settings    │ (Intent   │
│                  │ scattered)     │ scattered)   │ visible)  │
│                  │                │              │           │
│ Bias Risk        │ 🔴 High        │ 🔴 High      │ 🟢 Lower  │
│                  │ (Proxy         │ (Behavioral  │ (Explicit │
│                  │ discrimination)│ profiling)   │ constraints)
│                  │                │              │           │
│ GDPR Compliance  │ ⚠️ Requires    │ ⚠️ Requires  │ ✓ Native  │
│                  │ consent +      │ consent +    │ (No       │
│                  │ tracking       │ tracking     │ tracking) │
│                  │                │              │           │
│ Transparency     │ ⚠️ Low         │ ⚠️ Low       │ ✓ High    │
│                  │ (Black box)    │ (Black box)  │ (Rules    │
│                  │                │              │ inspectable)
│                  │                │              │           │
│ Ad Relevance     │ ✓ High         │ ✓ High       │ ✓ Good    │
│                  │ (via history)  │ (via history)│ (via      │
│                  │                │              │ intent)   │
│                  │                │              │           │
│ Publisher Revenue│ ✓ High         │ ✓ High       │ ✓ Good    │
│                  │ (Targeted ads) │ (Targeted)   │ (Fewer    │
│                  │                │              │ irrelevant)
│                  │                │              │           │
│ User Privacy     │ 🔴 Low         │ 🔴 Low       │ ✓ High    │
│                  │ (Tracked)      │ (Tracked)    │ (Not      │
│                  │                │              │ tracked)  │
│                  │                │              │           │
│ Regulatory Risk  │ 🟡 High        │ 🟡 High      │ ✓ Low     │
│                  │ (Fines, laws)  │ (Fines, laws)│ (Compliant)
│                  │                │              │           │
└──────────────────┴────────────────┴──────────────┴───────────┘
```

---

## IMPLEMENTATION ROADMAP

> **Note:** The following roadmap is a historical document outlining the original implementation plan.

```
┌──────────────────────────────────────────────────────────────┐
│           INTENT ENGINE LAUNCH TIMELINE                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Core Foundation (Months 1-2)                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ □ Design UniversalIntent schema                        │  │
│  │ □ Build intent extraction (text parsing)               │  │
│  │ □ Implement linguistic inference (temporal, skill)     │  │
│  │ □ Unit tests for extraction accuracy                   │  │
│  │ □ Schema validation library                            │  │
│  │                                                        │  │
│  │ Deliverable: TypeScript types + Python SDK             │  │
│  │ QA: 95%+ extraction accuracy on test queries           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Phase 2: Search Integration (Months 3-4)                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ □ Integrate intent extraction into Search              │  │
│  │ □ Implement constraint satisfaction                    │  │
│  │ □ Implement intent alignment scoring                   │  │
│  │ □ Test ranking accuracy (A/B vs. historical)           │  │
│  │ □ Benchmark performance (latency)                      │  │
│  │                                                        │  │
│  │ Deliverable: Search ranking using intent schema        │  │
│  │ QA: <50ms ranking latency, 90%+ user satisfaction      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Phase 3: Cross-Product Extension (Months 5-6)               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ □ Intent extraction for Docs, Mail, Calendar           │  │
│  │ □ Service recommendation algorithm                     │  │
│  │ □ Docs ranking using intent                            │  │
│  │ □ Mail organization via intent                         │  │
│  │ □ Calendar insights from intent                        │  │
│  │                                                        │  │
│  │ Deliverable: Intent engine supports 7 services         │  │
│  │ QA: Service recommendation accuracy 85%+               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Phase 4: Ad Matching (Months 7-8)                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ □ Implement ad matching algorithm                      │  │
│  │ □ Fairness constraints (no discrimination)             │  │
│  │ □ Test for bias (protected attributes)                 │  │
│  │ □ Differential privacy for metrics                     │  │
│  │ □ Advertiser fairness audit                            │  │
│  │                                                        │  │
│  │ Deliverable: Privacy-first ad matching                 │  │
│  │ QA: 0% biased ads, 100% fairness compliance            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Phase 5: Privacy & Compliance (Months 9-10)                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ □ TTL enforcement (8-hour session expiry)              │  │
│  │ □ Auto-deletion pipeline                               │  │
│  │ □ Encryption at rest (server logs)                     │  │
│  │ □ GDPR, CCPA, India DPDP audit                         │  │
│  │ □ Threat model validation                              │  │
│  │ □ Security review (external auditors)                  │  │
│  │                                                        │  │
│  │ Deliverable: Privacy-certified Intent Engine            │  │
│  │ QA: 100% GDPR/CCPA compliant, 0 privacy violations     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Phase 6: Production & Monitoring (Months 11-12)             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ □ Production deployment (staged rollout)               │  │
│  │ □ Monitoring dashboards (accuracy, quality)            │  │
│  │ □ User education (how intent works)                    │  │
│  │ □ Transparency reports (quarterly)                     │  │
│  │ □ User feedback loops & iteration                      │  │
│  │                                                        │  │
│  │ Deliverable: Intent Engine in production               │  │
│  │ QA: 99.9% uptime, <100ms latency, 95% satisfaction     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## KEY METRICS & SUCCESS CRITERIA

```
┌──────────────────────────────────────────────────────────────┐
│         HOW TO MEASURE INTENT ENGINE SUCCESS                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Extraction Accuracy (Intent schema completion)           │
│     ├─ Goal classification: 95%+ accuracy                   │
│     ├─ Constraint extraction: 90%+ precision/recall          │
│     ├─ Skill level detection: 85%+ accuracy                 │
│     └─ Use case classification: 88%+ accuracy               │
│                                                              │
│  2. Ranking Quality (User satisfaction)                      │
│     ├─ CTR (Click-through rate): >= baseline                │
│     ├─ Dwell time: >= baseline                              │
│     ├─ Query reformulation: < baseline                      │
│     ├─ Bounce rate: < baseline                              │
│     └─ User satisfaction survey: 4.0+ / 5.0                 │
│                                                              │
│  3. Privacy Compliance (Zero violations)                     │
│     ├─ Intent data deleted at session end: 100%             │
│     ├─ No persistent behavioral profiles: 100%              │
│     ├─ No cross-session user linking: 100%                  │
│     ├─ GDPR/CCPA audit: 100% compliant                      │
│     └─ Security audit: 0 critical vulnerabilities           │
│                                                              │
│  4. Fairness Metrics (No discrimination)                     │
│     ├─ Discriminatory ads rejected: 100%                    │
│     ├─ Equal ad impressions by group: p-value > 0.05        │
│     ├─ Fairness violation incidents: 0 per month            │
│     └─ Fairness audit pass rate: 100%                       │
│                                                              │
│  5. System Performance (Speed & reliability)                 │
│     ├─ Intent extraction latency: < 50ms                    │
│     ├─ Ranking latency: < 100ms                             │
│     ├─ Ad matching latency: < 75ms                          │
│     ├─ System uptime: 99.9%                                 │
│     └─ Error rate: < 0.1%                                   │
│                                                              │
│  6. Ad Relevance (Business metrics)                          │
│     ├─ CTR on matched ads: >= baseline                      │
│     ├─ Advertiser ROI: >= baseline                          │
│     ├─ False positive rate: < baseline                      │
│     └─ Publisher CPM: >= baseline                           │
│                                                              │
│  7. User Adoption (Product metrics)                          │
│     ├─ Users opting into intent inference: >= 50%           │
│     ├─ Users viewing intent signals: >= 40%                 │
│     ├─ Users customizing constraints: >= 20%                │
│     └─ Referral rate (recommend to friends): 4.2+/5.0       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## QUICK START FOR BUILDERS

### To Implement Intent Engine, You Need:

1. **Intent Extraction SDK** (Phase 1)
   - Language: Python/TypeScript
   - Size: ~2,000 LOC
   - Dependencies: None (no ML models, pure rules)

2. **Schema Validation** (Phase 1)
   - Validate intent objects against schema
   - Size: ~500 LOC
   - Test coverage: 100%

3. **Ranking Algorithm** (Phase 2)
   - Constraint satisfaction engine
   - Intent alignment scorer
   - Size: ~1,500 LOC
   - Benchmarked latency: <100ms

4. **Ad Matching Engine** (Phase 4)
   - Fairness constraint validator
   - Relevance scorer
   - Size: ~1,200 LOC
   - Zero bias guarantee

5. **Privacy/Compliance Layer** (Phase 5)
   - Session TTL enforcement
   - Auto-deletion pipeline
   - Encryption at rest
   - Size: ~800 LOC
   - 100% GDPR/CCPA compliance

**Total Effort**: ~12 months, 2-3 engineers, ~5-10k LOC

**Deployment**: On-premise or cloud (any provider)

---

## NEXT STEPS

1. **Review** this whitepaper + visual guide with your team
2. **Validate** feasibility with your use cases
3. **Prototype** intent extraction on sample queries
4. **A/B test** intent-based ranking vs. baseline
5. **Iterate** based on user feedback
6. **Launch** to production (staged rollout)
7. **Monitor** privacy, fairness, and quality metrics

---

**End of Visual Architecture Guide**

For detailed algorithms, see: [Intent-Engine-Whitepaper.md](Intent-Engine-Whitepaper.md)
