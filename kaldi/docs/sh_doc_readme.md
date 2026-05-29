# docker_baslat.sh walkthrough

This document explains each step in docker_baslat.sh in the order it runs. It is written for readers who are new to Kaldi, so it also explains what each common file and concept means.

## Quick glossary (very short)
- `wav.scp`: maps utterance IDs to audio file paths or commands.
- `text`: maps utterance IDs to transcripts.
- `utt2spk`: maps utterance IDs to speaker IDs.
- `MFCC`: compact acoustic features extracted from audio.
- `CMVN`: normalization stats to make features more consistent.
- `GMM-HMM`: classic acoustic model (Gaussian mixtures + HMMs).
- `WER`: word error rate, lower is better.

## File format examples (super small)
These are not from your data, just a simple example of the structure:

```
wav.scp
utt0001 /data/audio/utt0001.wav
utt0002 /data/audio/utt0002.wav

text
utt0001 merhaba dunya
utt0002 bugun hava guzel

utt2spk
utt0001 spk01
utt0002 spk01
```

## 0) Shell and banner
- `#!/bin/bash`: Run the script with Bash.
- `echo "=== Kaldi Turkce ASR Baslatiliyor ==="`: Print a start banner so you know the pipeline started.
  - What it does: just prints text, no data is changed.

## 1) Enter Kaldi recipe and load paths
- `cd /opt/kaldi/egs/commonvoice/s5`: Move into the Common Voice Turkish recipe directory. Kaldi recipes assume you run commands from this folder.
- `source ./path.sh`: Load Kaldi tool paths and environment variables for this recipe so tools like `steps/...` and `utils/...` are available.
  - What it does: makes sure the shell can find Kaldi scripts and binaries without full paths.

## 2) KenLM tools on PATH
- `export PATH=/opt/kaldi/tools/kenlm/build/bin:$PATH`: Ensure KenLM binaries (`lmplz`, `build_binary`) are available for language model training.
  - What it does: prepends the KenLM folder so the LM commands run from anywhere.

## 3) Parallelism setting
- `NJ=8`: Use 8 parallel jobs for steps that support `--nj`. Increase or decrease this depending on CPU cores and RAM.
  - What it does: sets a shell variable used by later commands to speed up processing.

## 4) Data preparation
- `python3 /data/proje/kaldi_data_prep.py`: Build Kaldi data directories (`tr_train`, `tr_dev`, `tr_test`) with `wav.scp`, `text`, `utt2spk`, and other files Kaldi needs.
- This step converts raw dataset files into Kaldi's standard format so later tools can read the data.
  - What it does: reads your dataset TSVs and creates the Kaldi data folders under `/data/kaldi_tr/`.

## 5) Sort data files
- For each of `/data/kaldi_tr/tr_train`, `/data/kaldi_tr/tr_dev`, `/data/kaldi_tr/tr_test`:
  - `sort .../utt2spk -o .../utt2spk`
  - `sort .../text -o .../text`
  - `sort .../wav.scp -o .../wav.scp`
- Kaldi expects these files to be sorted by utterance ID. Sorting avoids warnings and ensures reproducible results.
  - What it does: rewrites each file in sorted order, in-place, without changing content.

## 6) MFCC feature extraction
- `steps/make_mfcc.sh --nj $NJ --cmd "run.pl" ...`: Extract MFCC features for train, dev, test.
- The raw audio is turned into numeric feature matrices that models can learn from.
- Output goes into `/data/kaldi_tr/mfcc/*` and logs into `/data/kaldi_tr/log/*`.
  - What it does: runs feature extraction in parallel and stores `feats.scp` and feature archives.

## 7) CMVN stats
- `steps/compute_cmvn_stats.sh ...`: Compute cepstral mean and variance normalization stats for each split.
- This normalizes features so speakers and recording conditions are more consistent.
- Required for most GMM-HMM training and decoding steps.
  - What it does: generates `cmvn.scp` and updates data dirs with normalization info.

## 8) Lexicon and dictionary prep
- `python3 /data/proje/kaldi_lexicon_olustur.py`: Create `lexicon.txt` and related resources. A lexicon maps words to phone sequences.
- `python3 /data/proje/kaldi_dict_hazirla.py`: Create dictionary files in `/data/kaldi_tr/local/dict` (phones list, silence phones, etc.).
  - What it does: builds the pronunciation dictionary and phone inventories used by the language setup.

## 9) Language directory
- `utils/prepare_lang.sh /data/kaldi_tr/local/dict "<UNK>" /data/kaldi_tr/local/lang /data/kaldi_tr/lang`:
  - Builds the language FST input files (phones, words, L.fst, etc.).
  - This step combines the lexicon and phone inventories into a format used by decoders.
  - Uses `<UNK>` as the unknown word symbol.
  - What it does: creates the core `lang` directory used by graph building and decoding.

## 10) N-gram language model
- `awk '{$1=""; print $0}' .../tr_train/text > .../lm_train.txt`:
  - Remove the utterance ID column from `text`, keep only transcripts. The LM is trained on plain sentences.
  - What it does: produces a text-only corpus file for the language model.
- `lmplz -o 3 --text .../lm_train.txt --arpa .../lm.arpa`:
  - Train a 3-gram LM in ARPA format. A 3-gram uses the previous two words to predict the next.
  - What it does: estimates probabilities of word sequences from the training transcripts.
- `build_binary .../lm.arpa .../lm.bin`:
  - Build a binary LM for faster loading during decoding.
  - What it does: converts the text LM into a compact binary file.
- `gzip -k -f .../lm.arpa`:
  - Compress the ARPA file for `format_lm.sh`.
  - What it does: keeps both the `.arpa` and `.arpa.gz` versions.
- `utils/format_lm.sh /data/kaldi_tr/lang .../lm.arpa.gz .../lexicon.txt /data/kaldi_tr/lang_test`:
  - Create `lang_test` with the LM integrated, used for decoding graphs.
  - What it does: merges the LM with the lexicon and prepares the decode language folder.

## 11) Monophone training
- `steps/train_mono.sh --nj $NJ --cmd "run.pl" ...`:
  - Train a monophone GMM-HMM on the training split. This is the simplest acoustic model and a baseline.
  - Outputs to `/data/kaldi_tr/exp/mono`.
  - What it does: estimates a basic acoustic model to bootstrap later stages.

## 12) Monophone decoding
- `utils/mkgraph.sh .../lang_test .../exp/mono .../exp/mono/graph`:
  - Build the decoding graph for the monophone model. This combines the acoustic model, lexicon, and LM.
  - What it does: produces the `HCLG.fst` graph used during decoding.
- `steps/decode.sh --nj $NJ --cmd "run.pl" ...`:
  - Decode the test split and write lattices and scoring files.
  - What it does: runs the decoder and stores outputs under `decode_test`.
- `cat .../best_wer`:
  - Print the best WER for monophone.
  - What it does: shows the best scoring WER line for quick inspection.

## 13) Monophone alignment
- `steps/align_si.sh --nj $NJ --cmd "run.pl" ...`:
  - Align training data using the monophone model. Alignment assigns time boundaries to each phone.
  - Output alignments to `/data/kaldi_tr/exp/mono_ali`.
  - What it does: creates alignments used to train better context-dependent models.

## 14) Triphone (tri1) training
- `steps/train_deltas.sh --cmd "run.pl" 2000 10000 ...`:
  - Train a delta-feature triphone model (tri1). Triphones model context on both sides of a phone.
  - 2000 states, 10000 Gaussians.
  - What it does: trains a stronger model using context and delta features.

## 15) Triphone (tri1) decoding
- `utils/mkgraph.sh .../lang_test .../exp/tri1 .../exp/tri1/graph`:
  - Build the tri1 decoding graph.
- `steps/decode.sh ...`:
  - Decode the test split with tri1.
- `cat .../best_wer`:
  - Print tri1 WER.
  - What it does: lets you compare tri1 performance with the monophone baseline.

## 16) Triphone (tri1) alignment
- `steps/align_si.sh ... /data/kaldi_tr/exp/tri1_ali`:
  - Align training data with tri1. Better alignments usually improve the next model.
  - What it does: produces higher-quality alignments for tri2 training.

## 17) LDA+MLLT (tri2) training
- `steps/train_lda_mllt.sh --cmd "run.pl" 2500 15000 ...`:
  - Train tri2 with LDA+MLLT. These transforms improve feature representation for GMMs.
  - 2500 states, 15000 Gaussians.
  - What it does: learns feature transforms and a stronger acoustic model.

## 18) LDA+MLLT (tri2) decoding
- `utils/mkgraph.sh .../exp/tri2/graph`:
  - Build the tri2 decoding graph.
- `steps/decode.sh ...`:
  - Decode the test split with tri2.
- `cat .../best_wer`:
  - Print tri2 WER.
  - What it does: another checkpoint to see improvement from LDA+MLLT.

## 19) LDA+MLLT (tri2) alignment
- `steps/align_si.sh ... /data/kaldi_tr/exp/tri2_ali`:
  - Align training data with tri2 to prepare for SAT training.
  - What it does: produces alignments suitable for speaker adaptive training.

## 20) SAT (tri3) training
- `steps/train_sat.sh --cmd "run.pl" 4000 40000 ...`:
  - Train SAT (speaker adaptive training) tri3 model. It adapts to each speaker to reduce variability.
  - 4000 states, 40000 Gaussians.
  - What it does: trains the strongest GMM-HMM model in this pipeline.

## 21) SAT (tri3) decoding
- `utils/mkgraph.sh .../exp/tri3/graph`:
  - Build the tri3 decoding graph.
- `steps/decode_fmllr.sh ...`:
  - Decode with fMLLR adaptation, which is consistent with SAT training.
  - What it does: adapts features per speaker and decodes with the SAT model.
- `cat .../best_wer`:
  - Print tri3 WER.
  - What it does: final performance number for the GMM-HMM pipeline.

## 22) Final results summary
- Prints WERs for mono, tri1, tri2, tri3 from their respective decode directories.
- This makes it easy to compare progress across model stages.
  - What it does: gives a quick end-of-run summary without opening each directory.

## 23) Drop into shell
- `exec bash`: Replace the script process with an interactive shell so the container stays open and you can inspect outputs.
  - What it does: keeps the Docker container alive and puts you at a shell prompt.
