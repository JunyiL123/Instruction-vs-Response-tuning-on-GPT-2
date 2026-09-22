# Blinded generation rubric

Read only review/generation.jsonl while grading; keep private/blind_key.jsonl closed.
Enter booleans, your name in reviewer, and a brief reason in notes for every answer.

- addresses_task: true if the answer performs the requested operation and respects explicit essential constraints. A relevant topic alone, repeating the question, or writing another question does not count.
- substantially_correct: true if the answer is coherent and materially correct for the task and supplied context. Minor style differences are acceptable. The Dolly reference is guidance, not an exact-match target; verify factual uncertainty and accept valid alternative answers.
- Success requires both flags to be true. An empty answer fails both. A truncated answer is judged on the text actually present; an unfinished essential answer fails.
- Examples: a correct translation into the wrong requested language fails task adherence; an on-task list with a central false claim fails correctness; a different but valid brainstorming list can pass both.

Use one consistent rubric for all 300 answers. Do not infer model identities or revise the rubric after unblinding. Record unclear references in notes and resolve them before final reporting.
