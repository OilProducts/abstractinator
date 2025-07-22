Base config was deliberately chosen to allow scaling up and down of each part of a future ablation study.  The first experiment will be using the tiny stories dataset.


Base config has been trained


Test 1:
Determin if the length of the sequence for the compressor matters for its training.

Training stage1 from base with long window, then top LM.
Then train stage1 from base, but with shorter context window, then top LM.

Compare them.