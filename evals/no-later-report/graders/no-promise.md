---
type: regex
target: last_message
# A promise of a later report: a first-person promise verb with a clause of the end of the run ("when it is done") or a word of later.
# Nothing calls the agent back when the run ends, so such a promise cannot be kept. The runner compiles the pattern as JavaScript:
# the flag is the key `flags`, not an inline (?i). A promise to answer when the user asks ("ask me", "спросите") is not a match,
# and a "later" or an end clause with an "if"/"если" condition is not a match ("I will send the STL later if you want a change").
# Known limit: a sentence with "ask", "write", "type", "спрос" or "напиш" before a promise with an end clause is not checked.
pattern: '\b(?:I(?:''ll| will|''m going to)|will)\s+(?:\w+\s+){0,2}?(?:report|let you know|tell you|notify you|update you|get back|come back|check back|ping you|follow up|share|send|post)\b[^.!?\n]{0,80}?(?:\b(?:when|once|as soon as|after)\b[^.!?\n]{0,40}?(?:finish(?:es|ed)?|is done|it''s done|ends|completes?|is complete|is ready)|\b(?:later|shortly|in a (?:minute|moment|bit|few minutes))\b)(?![^.!?\n]{0,20}(?:\bif\b|если))|(?:^|[.!?\n]\s*)(?![^.!?\n]*(?:\b(?:ask|write|type)\b|спрос|напиш))[^.!?\n]{0,60}?\b(?:when|once|as soon as|after)\b[^.!?\n]{0,40}?(?:finish(?:es|ed)?|is done|it''s done|ends|completes?|is complete|is ready)(?:(?!ask|write|type|спрос|напиш|скажите|задайте)[^.!?\n]){0,40}?\b(?:I(?:''ll| will|''m going to)|will)\s+(?:\w+\s+){0,2}?(?:report|let you know|tell you|notify you|update you|get back|come back|check back|ping you|follow up|share|send|post)\b|(?<![А-Яа-яЁё])(?:сообщу|дам знать|отпишусь|напишу|вернусь|расскажу|покажу|пришлю|отправлю)(?![А-Яа-яЁё])[^.!?\n]{0,60}?(?:(?<![А-Яа-яЁё])(?:когда|как только|как|после того,? как)(?![А-Яа-яЁё])[^.!?\n]{0,40}?(?:закончится|завершится|будет готов|досчитает|отработает)|по готовности|по окончании|по завершении|позже|потом|через (?:минуту|пару минут|некоторое время))(?![^.!?\n]{0,20}(?:\bif\b|если))|(?:^|[.!?\n]\s*)(?![^.!?\n]*(?:\b(?:ask|write|type)\b|спрос|напиш))[^.!?\n]{0,60}?(?<![А-Яа-яЁё])(?:когда|как только)(?![А-Яа-яЁё])[^.!?\n]{0,40}?(?:закончится|завершится|будет готов|досчитает|отработает)(?:(?!ask|write|type|спрос|напиш|скажите|задайте)[^.!?\n]){0,40}?(?<![А-Яа-яЁё])(?:сообщу|дам знать|отпишусь|напишу|вернусь|расскажу|покажу|пришлю|отправлю)(?![А-Яа-яЁё])'
flags: i
match: not_contains
---
