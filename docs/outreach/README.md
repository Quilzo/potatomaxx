# Outreach drafts

Ready-to-send materials, in the order that makes sense.

## Sent

| Date | Where | Link |
|---|---|---|
| 2026-08-22 | llama.cpp Discussions, *Show and tell* | [#27534](https://github.com/ggml-org/llama.cpp/discussions/27534) |

## Not yet sent

Each remaining item goes out in the maintainer's own name, and mailing-list
archives are permanent.

The sequencing is deliberate: foundations ask "who uses this?", and right now the
honest answer is nobody. These are how that changes.

## Order, and why

| # | Where | What it asks for | Why first |
|---|---|---|---|
| 1 | [`io-uring-list.txt`](io-uring-list.txt) → `io-uring@vger.kernel.org` | Whether the batch-vs-sliding-window result and the loss to a thread pool are expected | Genuine technical question with data, to the people best placed to answer. Costs nothing, and the answer improves the code either way. |
| 2 | [`llama-cpp.md`](llama-cpp.md) → llama.cpp Discussions, *Show and tell* | A sanity check on the correctness argument, and a router-trace dump | The users are there, and the repack is a drop-in GGUF so it needs nothing from them. Also where a routing trace can realistically come from. |
| 3 | [`localllama.md`](localllama.md) → r/LocalLLaMA | People with real MoE models and varied hardware to run `analyse` and `kio` | Broadest reach for the one thing most needed: real-hardware numbers. |
| 4 | [`linux-mm.txt`](linux-mm.txt) → `linux-mm@kvack.org` | Whether per-range page-cache retention is expressible, or whether userspace caching is the intended answer | Send *after* io-uring; a narrower question, and better asked once the storage numbers have survived scrutiny. |
| 5 | [`gnu-questionnaire.txt`](gnu-questionnaire.txt) → `gnueval@gnu.org` | GNU package evaluation | Send once there is at least one real-model result. GNU accepts unfinished work, but "never run on a real model" is a fair objection and cheap to remove. |

## How to send the mailing-list posts

Both `.txt` drafts are already plain text, wrapped at 74 columns, with `To:` and
`Subject:` as the first two lines. They need to go out from a **personal**
address, not a work one: vger and kvack archives are public and permanent
(lore.kernel.org), so the `From:` header is published for good.

Easiest, using `git send-email` — it will not mangle the formatting the way a
webmail composer can:

```bash
git config --global sendemail.smtpServer smtp.gmail.com
git config --global sendemail.smtpServerPort 587
git config --global sendemail.smtpEncryption tls
git config --global sendemail.smtpUser you@gmail.com   # app password, not your login password
git send-email --from="Your Name <you@gmail.com>" docs/outreach/io-uring-list.txt
```

Gmail needs an [app password](https://myaccount.google.com/apppasswords) for SMTP.

By hand, if you prefer: open Gmail logged in as the personal account, paste the
body below the `To:`/`Subject:` lines, and **turn off rich formatting** (the
`Aa` / "plain text mode" toggle) before sending. Kernel lists silently drop HTML
mail.

Subscribe first, or replies will be missed:
- <https://subspace.kernel.org/vger.kernel.org.html> for `io-uring`
- <https://kvack.org/majordomo/> for `linux-mm`

## Before sending any of it

- **Bare metal is not available to the author.** Every I/O number was measured
  under WSL2, where the block path is virtualised. Rather than hide that, each
  draft states it plainly and asks the reader to reproduce or refute — which is a
  better use of a mailing list than a claim would be. `potatomaxx kio` prints the
  table in about a minute with no model, GPU or configuration, which makes the ask
  cheap to honour.

- **Do not send the mailing-list posts from a work address.** vger and kvack
  archives are public and permanent (lore.kernel.org), so the `From:` header is
  effectively published for good. Send from a personal address.
- **Kernel lists want plain text**, no HTML, no top-posting, wrapped at ~75
  columns. Both `.txt` drafts are already wrapped and formatted for that; send
  them with `git send-email` or a client set to plain text, and do not let a
  webmail client reflow them.
- **Subscribe before posting** to `io-uring@vger.kernel.org` and
  `linux-mm@kvack.org`, or replies will be missed.
- **`gnu-questionnaire.txt` has one bracketed item left**: whether you agree to
  follow GNU policies. That is a commitment to GNU maintainership — an ongoing
  obligation, not a formality. See [`../SUBMISSION.md`](../SUBMISSION.md) for what
  it entails.

## What not to do

- Don't lead with the tool. Every draft leads with a measurement or a question,
  because that is what these audiences respond to and what is actually useful to
  them.
- Don't drop the caveats to make the numbers look better. The WSL2 caveat, the
  "never run on a real model" admission, and the findings that contradicted the
  design are the reason any of this is worth reading. They are also what makes a
  correction easy to accept rather than embarrassing.
- Don't quote the projected figures as measured. `SUBMISSION.md` and the README
  both distinguish measured, computed and unknown; keep that distinction in any
  reply.
