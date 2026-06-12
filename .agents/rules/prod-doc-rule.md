---
trigger: always_on
---

# Project Intelligence Rule

You are responsible for creating and maintaining `PROJECT_CONTEXT.md` —
a living SDLC document that captures the complete lifecycle, thinking, and
decision-making of this project. It is NOT a README. It is NOT a changelog.
It is the permanent record of how and why this project came to exist, how
every decision was made, and what was explored — including directions that
were tried and abandoned. Written as a meticulous senior engineer who
documents not just what was chosen, but what was considered, what failed,
and what was learned.

---

## On First Message of Any New Project

If `PROJECT_CONTEXT.md` does not exist in the project root:

1. Create it immediately using the Full Template at the bottom of this rule.
2. Fill in every section you can infer from context.
3. Leave unfilled sections with a `[TBD]` placeholder.
4. Tell the user: "I've created PROJECT_CONTEXT.md to document this
   project's full lifecycle. I'll keep it updated automatically as we work."
5. Ask ONE founding question: "To start — what problem are you solving
   and who has it?"
6. Update the document with their answer before proceeding with any task.

---

## On Every Message — Before Starting Work

1. Read `PROJECT_CONTEXT.md` in full.
2. Understand all prior decisions, constraints, architecture, and — critically
   — every exploration that was tried and abandoned, before acting.
3. Never re-propose an approach that was already tried and failed without
   explicitly acknowledging it was tried, what happened, and why this new
   attempt is different.
4. Never contradict a prior decision without documenting the reversal,
   what changed, and what was learned from the original direction.
5. Capture any thinking or intent the user expresses in this message, even
   casually. These belong in § Conversations & Thinking.

---

## On Every Message — After Completing Work

### The Core Rule: Preserve History, Update Facts

Not all sections work the same way. Apply the right update strategy per section:

---

### STRATEGY A — Overwrite / Replace (factual current state)

These sections describe what is true RIGHT NOW. Replace outdated content.
But: when replacing, if the old state is meaningfully different from the new,
note the change inline — e.g., "Previously X. Changed to Y on [date] because Z."

Sections:
- Current Status
- Active Work
- Open Questions (remove when resolved — move to Design Decisions)
- Tech Stack table entries

---

### STRATEGY B — Preserve + Contextualize (exploration history)

These sections must NEVER have entries deleted. When something changes,
update the entry to reflect what happened — adding the outcome, the reversal,
or the lesson — rather than removing it.

Specific rules:

**Design Decisions (ADRs)**
- Never delete an ADR.
- If a decision is reversed, mark it: `**Status**: Superseded — [date]`
- Add a "Reversal" block to the original ADR (see format below).
- Create a NEW ADR for the replacement decision that references the original.
- The old ADR stays in the document so readers understand the full arc.

**Known Issues**
- Never delete an issue entry.
- When resolved: update Status to `Resolved [date]` and add a "Resolution" line.
- Pattern: resolved issues are proof of work and context for future bugs.

**Experiments & Spikes / Exploration Journal**
- Always additive. Never delete.
- When an explored direction fails: document what was tried, what happened,
  and what the conclusion was. This is valuable precisely because it failed.
- When a direction is revisited: add a new entry referencing the original,
  noting what changed that made it worth trying again.

**Assumptions Log**
- When an assumption is validated: mark it `Validated [date]` and note what
  confirmed it.
- When an assumption is invalidated: mark it `INVALIDATED [date]`, note what
  disproved it, and add an entry to Lessons Learned.

**Lessons Learned**
- Always additive. Never delete or overwrite.

**Conversations & Thinking**
- Always additive. Most recent entries at the top.
- When the section exceeds ~15 entries: compress older entries into a
  paragraph summary, keeping full detail only for the most recent 8-10.

---

### STRATEGY C — Evolve with narrative (architecture, data model)

These sections describe the current design but may have changed over time.

- Update the main content to reflect the current state.
- If the change is significant (e.g., switched databases, restructured service
  boundaries), add a brief `> Previously: [X]. Changed [date] because [Y].`
  callout directly below the changed content.
- Do not narrate every small change — only pivots that matter.

---

## What to Capture and Where

| What happened | Section | Strategy |
|---|---|---|
| User expressed a goal, concern, opinion, or idea | § Conversations & Thinking | B |
| Model proposed an approach and explained why | § Conversations & Thinking | B |
| Architectural choice made | § Architecture + § Design Decisions | B + C |
| Technology chosen | § Tech Stack + § Design Decisions | A + B |
| Requirement clarified or added | § What We're Building | C |
| Requirement explicitly cut | § Explicit Non-Goals | C |
| Design decision made | § Design Decisions (full ADR) | B |
| Decision reversed or superseded | § Design Decisions (reversal block) | B |
| Feature or approach tried and abandoned | § Exploration Journal | B |
| Bug found | § Known Issues | B |
| Bug resolved | § Known Issues (update entry) | B |
| Something failed or experiment concluded | § Exploration Journal + § Lessons Learned | B |
| Something surprising discovered | § Lessons Learned | B |
| Assumption made | § Assumption Log | B |
| Assumption validated or disproved | § Assumption Log (update entry) | B |
| Risk identified | § Risk Register | B |
| Risk resolved/changed | § Risk Register (update entry) | B |
| Unresolved question | § Open Questions | A |
| Question resolved | Move to § Design Decisions, remove from Open Questions | A |
| Future work identified | § Roadmap | A |
| Standard or pattern established | § Engineering Standards | C |
| NFR defined or target set | § Non-Functional Requirements | C |

---

## Key Formats

### ADR — Standard decision

```
## [Decision Name]

**Date**: YYYY-MM-DD
**Status**: Decided

**Context**:
What situation made this decision necessary?

**Decision**:
What was decided?

**Reasoning**:
Why this choice? What specific factors drove it?

**Alternatives Considered**:
- [Option A] — rejected because [reason]
- [Option B] — rejected because [reason]

**Explicit Non-Decision**:
[What did we consciously decide NOT to do, and why? This is as important
as what we decided to do.]

**Trade-offs**:
What does this cost us? What risks does it carry?

**Consequences**:
What does this constrain or change going forward?
```

### ADR — Reversal block (append to original ADR when superseded)

```
---
**⚠ Superseded**: YYYY-MM-DD

**Why reversed**:
What changed that made the original decision wrong or no longer appropriate?

**What we learned from this direction**:
What did the original approach teach us, even if it didn't work out?

**What's preserved**:
Any part of the original thinking still valid in the new approach?

**See**: [New ADR name]
---
```

### Exploration Journal entry

```
## [What was explored] — [Date]

**Why we tried it**:
What problem were we trying to solve? What made this approach seem promising?

**What we did**:
Concise description of what was actually attempted.

**What happened**:
Outcome. Be honest. "It didn't work" is a valid and useful outcome.

**Why it didn't work** (if applicable):
Root cause, not just symptoms.

**What we learned**:
What does this tell us that we didn't know before?

**Impact on direction**:
Did this change our approach? Did we go back to something? Did we pivot?

**Revisit?**:
Is there a condition under which this would be worth trying again?
```

### Conversations & Thinking entry

```
## [Topic] — [Date]

**User's intent / thinking**:
What the user was trying to accomplish. Their concerns, preferences,
constraints — including informal or passing comments that signal direction.

**Model's analysis**:
What was reasoned through. What the model considered before acting.
What was recommended and why.

**What was explored**:
Approaches, alternatives, or ideas that came up in this conversation,
even if not chosen.

**Outcome**:
What was decided or done.
```

---

## Maintenance Rules

- Keep `PROJECT_CONTEXT.md` under 600 lines (the extra headroom is for
  preserved history, which is intentional).
- When approaching the limit: compress Conversations & Thinking (older
  entries → summary paragraph). Do NOT compress Design Decisions,
  Exploration Journal, or Lessons Learned — these must stay intact.
- Remove `[TBD]` placeholders as they are filled in.
- Update `Last Updated` and `Current Phase` on every edit.
- Prefer writing for a senior engineer joining the project in 6 months
  who needs to understand not just the current state, but the reasoning
  and the journey to get here.

---

## Full Template

```markdown
# [Project Name] — Project Intelligence Document

> This is the living SDLC record of this project. It captures not just
> what was built, but why every decision was made, what was explored and
> abandoned, what failed and what was learned. Written so that any future
> engineer or AI model can achieve full context from this document alone.

**Started**: [Date]
**Last Updated**: [Date]
**Current Phase**: Ideation / Planning / Development / Testing / Launch / Maintenance
**Status**: [One sentence on where things stand right now]

---

## 1. Why This Exists — The Problem

**Problem Statement**:
[What specific problem does this solve? Be precise. What breaks without it?]

**Who has this problem**:
[Who are they? What's their context? Why is this painful for them specifically?]

**Current state / workarounds**:
[How do they handle it today? Why is that inadequate?]

**Why now**:
[What makes this the right time to build this?]

---

## 2. What We're Building

**Vision**:
[One sentence: what does this become at its best?]

**Core value proposition**:
[What does the user get that they cannot get elsewhere?]

**In scope**:
-

**Explicit Non-Goals** (out of scope — and why):
[This is as important as scope. What are we consciously NOT building, and
what is the reasoning? This prevents scope creep and re-litigation.]
-

**Success looks like**:
[How do we know this worked? What is the measurable signal?]

---

## 3. Who It's For

**Primary users**:
[Who uses this directly? What do they know? What do they need?]

**Secondary users / stakeholders**:
[Who else is affected?]

**User assumptions**:
[What do we assume about our users' technical level, context, behavior?]

---

## 4. Non-Functional Requirements

[These drive architecture decisions as much as functional requirements do.
Document them explicitly so they can be defended later.]

| Requirement | Target | Rationale |
|---|---|---|
| Response time | | |
| Uptime / availability | | |
| Concurrent users | | |
| Data retention | | |
| Security classification | | |
| Compliance | | |

---

## 5. Architecture

**System overview**:
[How does the system work at a high level? How do the parts connect?]

**Component breakdown**:
-

**Data flow**:
[How does data move through the system from input to output?]

**Key boundaries**:
[Hard lines between components. What must stay separate and why.]

---

## 6. Tech Stack

| Layer | Technology | Why this choice | Risk |
|---|---|---|---|
| Frontend | | | |
| Backend | | | |
| Database | | | |
| Auth | | | |
| Infrastructure | | | |
| Testing | | | |

---

## 7. Data Model

**Key entities**:
[Core objects in the system.]

**Relationships**:
[How they relate.]

**Constraints**:
[Business rules enforced at the data layer.]

---

## 8. API & Interfaces

**External APIs consumed**:
-

**Internal API contracts**:
[Key endpoints or interfaces other parts depend on.]

**Integration points**:
[Where this system touches other systems.]

---

## 9. Security Model

**Sensitive data**:
[What data is sensitive? What classification?]

**Threat surface**:
[Where could this system be attacked or abused?]

**Auth & authorization design**:
[Who can do what, and how is that enforced?]

**Security constraints that must never be violated**:
-

---

## 10. Design Decisions

[ADR entries. Never delete — supersede with reversal blocks.]

---

## 11. Engineering Standards

**Coding conventions**:
-

**Patterns in use**:
-

**Testing approach**:
-

**Deployment process**:
-

**Things that must never be violated**:
-

---

## 12. Assumption Log

[Assumptions we're operating on that haven't been validated.
Track whether they've been confirmed or disproved.]

| Assumption | Type | Status | Evidence |
|---|---|---|---|
| [What we're assuming] | Technical / User / Business | Unvalidated / Validated [date] / INVALIDATED [date] | [What confirmed or disproved it] |

---

## 13. Risk Register

[What could go wrong. Documented before it happens.]

| Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|
| | | | | |

---

## 14. Conversations & Thinking

[Running record of user thinking and model reasoning. Most recent first.
Compress entries older than ~10 into a summary paragraph when needed.]

---

## 15. Exploration Journal

[Every direction tried — including the ones that didn't work out.
This section is additive only. Never delete entries.]

---

## 16. Known Issues & Technical Debt

[Never delete entries. Update status in place.]

| Issue | Severity | Status | Workaround | Resolution (if any) |
|---|---|---|---|---|
| | | | | |

---

## 17. Lessons Learned

[Additive only. Mistakes, surprises, things worth remembering.]

**Mistakes made**:
-

**Surprising discoveries**:
-

**What we'd do differently**:
-

---

## 18. Open Questions

[Unresolved decisions. When resolved: move to Design Decisions, remove here.]

-

---

## 19. Roadmap

**Next up** (committed):
-

**Backlog** (likely):
-

**Abandoned** (tried or planned, then cut — with reason):
-

**Ideas** (maybe someday):
-

---

## 20. References

[Links, docs, research, and inspiration that informed decisions.]

-
```
