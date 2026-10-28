# Submission format

## 1. What you submit

One `.npz` per machine config, keyed **per shot in Hugging Face stream order** (0-indexed).
For each shot, `T` = number of `efit_times`:

| Key | Shape | Meaning |
|---|---|---|
| `shot_XXXX_psirz` | `(T, 65, 65)` | flux map `ψ(R,Z)` at each timestep (both machines 65×65) |
| `shot_XXXX_q95` | `(T,)` | edge safety factor |
| `shot_XXXX_betaN` | `(T,)` | normalized beta |

```
diii_d_public_test.npz
mast_public_test.npz
```

In the **Final** phase the names become `diii_d_private_test.npz` / `mast_private_test.npz`. The
scorer finds your predictions **by filename**, so these must be exact.

**That is the whole contract — the flux map plus the only two scalars a flux map cannot contain.**
You do **not** submit the LCFS, `R_axis`/`Z_axis`, `li`, or any shape scalar: the scorer derives all
of them from your submitted `psirz` with the same published functionals it applies to the ground
truth (see **Evaluation**). A scalar can only be earned by a flux map that implies it. Extra keys
are ignored, and predictions must be finite for DIII-D.

> **Do not round, truncate, decimate, or drop frames.** `np.round(psi, 3)` looks free — it leaves
> R²ψ at 0.999965 — while destroying **35% of the MAST Consistency term**, because `Consistency`
> compares shape scalars derived from your ψ and MAST's triangularity varies over a narrow range.
> `float16` is safe and is what the starter kit writes; fixed-decimal rounding is not.

## 2. Build it

```bash
python submission_skeleton.py --max-shots 0 \
    --repo your-username/fusion-eq-predictions \
    --read-token hf_...
```

That one command builds every shot, validates the structure, pushes to Hugging Face, and writes
`submission_pointer.zip` — the file you upload here. `--max-shots 0` means every shot; **the
default is 5**, a fast format check that would score almost everything as missing.

A full run takes a while and is not stuck: it streams the public test fold once to build, again to
validate, and only then uploads.

Run it once *without* `--read-token` first: it creates the repo and prints how to scope a token to
it. Hugging Face cannot scope a token to a repo that does not exist yet.

## 3. Submit it

Upload `submission_pointer.zip` on the **Submit** tab. It is a few hundred bytes — your predictions
travel to the scorer from Hugging Face, not through Codabench.

Measured from the scoring machine: **Codabench's file storage sustains ~0.5 MB/s, the Hugging Face
CDN ~50 MB/s.** A ~2 GB direct upload spends about an hour in transfer before scoring starts; the
pointer takes under a minute. The scoring worker runs one job at a time, so that hour is queue time
everyone shares.

Your predictions repo is **private** — nothing you submit is visible to other teams.

### The two tokens

| | what it does | where it goes |
|---|---|---|
| **write** token | uploads your `.npz` | `huggingface-cli login`, never leaves your machine |
| **fine-grained read** token | lets the scorer read your private repo | inside `manifest.json`, submitted |

Create the read token at
**[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)** → *New token* →
**Fine-grained** → under *Repository permissions* select your predictions repo → tick **only**
"Read access to contents of selected repos". The starter kit **refuses** a write token or a classic
read token there: both grant far more than the scorer needs, and that token is stored with your
submission. Revoke it when the competition ends.

### If you are not using the starter kit

The zip contains one file, `manifest.json`, at the archive root:

```json
{"repo_id": "your-username/fusion-eq-predictions",
 "revision": "0123456789abcdef0123456789abcdef01234567",
 "token": "hf_..."}
```

Each of these is enforced by the scorer with an explicit error message:

- It must be a **dataset** repo, and the token must be able to read it. (A public repo also works,
  with no token — but private is recommended.)
- `revision` must be the **40-character commit SHA**, not `main` or a tag. A moving reference would
  let a submission change after it was scored, and the reproduction check in the Rules depends on it
  not being able to.
- The `.npz` must sit at the **root** of the repo, named exactly as in section 1.

Because the commit is pinned, you can keep pushing to the same repo; each submission names the
commit it was built from.

## 4. Final phase: the reproduction package

**Every Final-phase submission must be accompanied, at submission time, by a reproduction package
reference.** This is new as of 2026-09-23 and applies to the Final phase only — the Development
phase is unchanged. Put the reference in the **Code URL** field of the submission fact sheet, as
either:

- a **Git repository at a pinned 40-character commit SHA**, on any public forge — GitHub, GitLab,
  Codeberg, an institutional host; no preference, and GitHub is not required:

  ```
  https://gitlab.com/yourteam/fusion-eq@3f2a91c4d8e5b7061a2c9f4e8d3b5a7c9e1f2d40
  ```

- or a **versioned archive on any public host** (Zenodo, Hugging Face, institutional storage), with
  its **SHA256** after the URL:

  ```
  https://zenodo.org/records/1234567/files/fusion-eq-final.tar.gz sha256:9f86d081884c7d65...
  ```

`Code URL` stays optional in the Development phase — fill it if you like, but nothing depends on it
there. It is required for Final-phase submissions.

The package must contain:

| | |
|---|---|
| **Source** | the model code that produced the submission |
| **Environment** | a pinned specification (lockfile, `environment.yml`, or a container image) |
| **Weights** | the trained weights used for this submission |
| **Training entrypoint** | one command that rebuilds the model **from scratch** |
| **Inference entrypoint** | one command that regenerates the submitted `.npz` from those weights |
| **Data provenance** | a short statement, **per machine**, of every external dataset and pre-trained model used |

Declare your training compute budget (hardware and wall-clock). Training must be reproducible
within **24 GPU-hours on a single 80 GB A100-class GPU**, or equivalent CPU time. If your pipeline
genuinely cannot fit that ceiling, say so in the package rather than omitting it and we will
arrange a case-by-case check.

**What we do with it.** We rebuild the **top five entries per challenge** — retrain from your
training entrypoint, re-score the result, and separately check that your shipped weights reproduce
your submitted predictions. An entry that cannot be rebuilt stays on the leaderboard but is not
eligible for an award. The binding prize condition is unchanged: the **top three** teams per award
category release source under an OSI-approved licence and file a 1–2 page methods report before
prizes are paid.

A container image is welcome but not required. If you already submit through the pointer route, the
pinned commit you are using satisfies the repository form of this requirement — add the weights, the
two entrypoints and the provenance statement to that same repo.

## 5. Limits

| | |
|---|---|
| **Development** | 5 submissions/day, 100 total, until **October 18, 2026** |
| **Final** | 3 submissions total, October 19–26, 2026 — blind, leaderboard hidden until close |

One submission enters both challenges: **Challenge 1** needs only the DIII-D file, **Challenge 2**
(`G_ratio`) needs both.

**Submitting predictions you did not produce is plagiarism and disqualifies the whole team** — see
the Rules page.

<details>
<summary>Fallback: uploading the .npz directly</summary>

If you would rather not use Hugging Face, zip the `.npz` at the **root** of an archive and upload
that instead. It scores identically.

```bash
python -c "import zipfile,pathlib; \
z=zipfile.ZipFile('submission.zip','w'); \
[z.write(p, p.name) for p in pathlib.Path('submission').glob('*.npz')]; z.close()"
```

Expect roughly an hour of transfer per submission, and note it counts against your **15 GB**
Codabench storage quota — use the **Resources** tab to delete superseded submissions, since the
leaderboard keeps only your last one pinned (`Force_Last`). The pointer route costs ~0 quota.

Do not compress the `.npz` themselves any further: `np.savez_compressed` already produces a DEFLATE
zip, so re-wrapping gains ~0% and may break the reader.
</details>