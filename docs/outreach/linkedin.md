<!-- LinkedIn post. The hook is the negative result, because it is both the most
     interesting thing here and the most defensible. Every number below is
     measured and sourced in the repository. -->

# Main version

I built a tool to make large AI models run on cheap hardware.

Then I measured it properly, and the core idea didn't work.

I shipped it anyway — and that turned out to be the point. 👇

Mixture-of-Experts models, the architecture behind many of the largest open LLMs, are too big to fit in RAM. So they stream from disk while running. That makes them a **storage** problem, not a compute problem.

My idea: reorder the experts on disk so the ones used together sit next to each other. Fewer, larger reads. Obviously a win.

It isn't. I surveyed seven real MoE checkpoints — Qwen3, DeepSeek, Mixtral, Granite, OLMoE — and **not one** has expert matrices small enough for reordering to matter. The smallest is 288 KB against a ~256 KB threshold. DeepSeek-V2-Lite is 1.5 MB.

Correct, provably safe, and useless on every model that currently exists.

But measuring it found what does matter:

**→ Queue depth is worth 33x.** Same NVMe drive: 0.099 GB/s reading one block at a time, 3.29 GB/s with 16 in flight. Most inference engines leave this on the table.

**→ LFU caching is worse than LRU** for experts — 78% vs 81% of the theoretical optimum. A cost-aware policy reaches 91%. (An open llama.cpp feature request currently proposes LFU.)

**→ Compressing quantised weights buys 2%,** not the 3x the papers imply — because a good quantiser has already spent that redundancy. On float weights it's 28%. Same code, same file, two different answers.

**→ Under 0.5% of experts are load-bearing.** Prune one and you lose 21–27% accuracy; reasoning collapses entirely. And they can be *rarely used* — so any frequency-based optimisation targets them first, and their own error metrics look completely normal. The tool now refuses to compress anything until you've identified them.

So it became something more useful than a fast thing: it tells you which optimisations are worth attempting on **your** model and **your** hardware, and which measurably are not.

Four of the ideas it now rejects are implemented unconditionally in comparable tools. Two of them were mine.

Rust, zero dependencies, 171 tests, GPL-2.0. Validated on a real checkpoint: 782 MB of weights repacked and verified bit-identical, and both quantisation decoders checked against ground truth.

**github.com/Quilzo/potatomaxx**

The lesson I keep relearning: a tool that only reports wins can't be trusted when it reports one.

#MachineLearning #Rust #SystemsEngineering #LLM #OpenSource #AI #Performance

---

# Short version (if you want something punchier)

I built a tool to run big AI models on cheap hardware. Then I measured it, and my core idea didn't work.

Mixture-of-Experts models stream from disk because they don't fit in RAM. My plan was to reorder experts on disk so co-used ones sit together — fewer, bigger reads.

I surveyed seven real checkpoints (Qwen3, DeepSeek, Mixtral, Granite). Not one has expert matrices small enough for it to matter.

What the measurements *did* find:

→ Queue depth is worth **33x** on the same drive — 0.099 vs 3.29 GB/s
→ LFU expert caching is **worse than LRU**; cost-aware beats both
→ Compressing quantised weights buys **2%**, not the 3x papers imply
→ **Under 0.5% of experts** are load-bearing — lose one, lose 21–27% accuracy — and they're often *rarely used*, so frequency-based optimisation targets them first

So the tool's job changed. It now tells you which optimisations are worth attempting on your model and hardware, and which measurably aren't. Four of the things it rejects are shipped unconditionally elsewhere. Two were mine.

Rust, zero dependencies, 171 tests, GPL-2.0.
**github.com/Quilzo/potatomaxx**

A tool that only reports wins can't be trusted when it reports one.

#MachineLearning #Rust #LLM #OpenSource #AI
