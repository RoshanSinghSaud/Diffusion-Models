# Diffusion Model (DDPM) from Scratch

A minimal, from-scratch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** in PyTorch.

## Idea in one line

Learn to turn pure noise into a realistic image by training a model to remove noise, one small step at a time.

## Intuition

Think of every image as a single point in a very high-dimensional space (one dimension per pixel). The images in a dataset don't fill this space randomly — they cluster together and form a structured surface (a **manifold**). Generating a new image means landing on a new point on that surface.

Diffusion works in two directions:

- **Forward:** slowly add noise to a real image until it becomes pure random noise.
- **Reverse:** train a neural network to undo that noise, step by step, so we can start from pure noise and walk back to a clean image.

## Forward process (adding noise)

We gradually corrupt an image `x_0` over `T` steps. The important trick: we don't *only* add noise — we also shrink the image a little at each step. This keeps the total variance stable, so after enough steps the image becomes clean Gaussian noise `N(0, I)`.

Because every step is Gaussian, we can jump straight to any step `t` with one formula (no loop needed):

```
x_t = √(ᾱ_t) · x_0 + √(1 − ᾱ_t) · ε        ε ~ N(0, I)
```

- `β_t` — how much noise we add at step `t` (the **noise schedule**)
- `α_t = 1 − β_t`
- `ᾱ_t` — the product of all `α` up to step `t`

The forward process has **no trainable parameters** — it is completely fixed.

## Reverse process (removing noise)

In the denoising process we have a noisy image `x_t`, and we want to predict a slightly cleaner image one step back, `x_{t-1}`.

What we'd *like* is the reverse posterior `q(x_{t-1} | x_t)`. But on its own this is intractable: to undo one step of noise you'd need to know where the image started, and a single noisy `x_t` could have come from a huge number of different original images. Without that anchor, the reverse step is ambiguous.

The fix is to also condition on the original image `x_0`. Then the posterior becomes tractable, with a clean closed form:

```
q(x_{t-1} | x_t, x_0)      ← exact, closed-form Gaussian
```

But here's the catch: this true posterior is **exact yet useless for generation**, because it needs `x_0` — the very thing we're trying to create. At sampling time we start from pure noise and have no `x_0`.

So DDPM defines a **second distribution**, the model posterior, which does *not* need `x_0`:

```
p_θ(x_{t-1} | x_t)         ← approximated by a neural network
```

This closes the loop with training and sampling:

- During **training** we still have `x_0` (it comes from the dataset), so we can compute the true posterior `q(x_{t-1} | x_t, x_0)` and train `p_θ` to match it.
- During **sampling** we don't have `x_0`, so we rely on the trained `p_θ` alone.

## Training

Training is surprisingly simple:

1. Take a real image `x_0`.
2. Pick a random step `t` and random noise `ε`.
3. Build `x_t` using the forward formula.
4. Ask the network to predict `ε` from `x_t` and `t`.
5. Loss = mean squared error between the true `ε` and the predicted `ε`.

```
Loss = ‖ ε − ε_θ(x_t, t) ‖²
```

That's the whole objective — predict the noise, minimize MSE.

## Sampling (generating images)

Start from pure noise `x_T ~ N(0, I)` and denoise step by step down to `x_0`:

```
x_{t-1} = (1/√α_t) · ( x_t − (β_t / √(1 − ᾱ_t)) · ε_θ(x_t, t) ) + σ_t · z
```

where `z ~ N(0, I)` for `t > 1`, and `z = 0` on the final step.

Two details that matter:

- We **don't remove all the noise at once.** We subtract a small, scaled amount each step, which keeps the walk stable.
- We **add a little fresh noise back** (`σ_t · z`) each step. This pulls off-track points back toward realistic images. (Same idea as Langevin sampling / score-based models.)

## Model architecture (U-Net)

The noise predictor is a **U-Net**:

- **Downsampling path** — spatial size shrinks, channels grow.
- **Bottleneck** — spatial size and channels stay constant.
- **Upsampling path** — spatial size grows, channels shrink.
- **Skip connections** — each downsampling layer connects directly to the matching upsampling layer, so fine detail isn't lost in the bottleneck.
- **Self-attention** at lower resolutions lets the model use global context.

The timestep `t` is turned into a **sinusoidal embedding** and fed into each block, so the network always knows how noisy the current image is.

## Project structure

```
.
├── model.py        # U-Net + time embedding
├── diffusion.py    # forward noising + sampling loop
├── train.py        # training loop
├── sample.py       # generate images from a trained model
└── README.md
```
*(adjust to match your actual files)*

## Usage

```bash
# install dependencies
pip install -r requirements.txt

# train
python train.py

# generate images
python sample.py
```

## References

- Ho et al. — *Denoising Diffusion Probabilistic Models* (2020)
- Nichol & Dhariwal — *Improved DDPM* (2021) — cosine noise schedule
- Song et al. — *Score-Based Generative Modeling* — the score / Langevin connection
