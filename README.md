# Contact Map Diffusion (Baseline)

An end-to-end PyTorch pipeline that trains a denoising diffusion model (DDPM-style) on protein Cα distance maps. The UNet learns to predict the noise added to a map at a random timestep, and sampling then removes noise step by step to generate new maps. The pipeline then reconstructs a 3D Cα trace from a generated map using metric MDS.

Status: Work in progress. The pipeline runs end to end, but the generated samples are not valid protein structures yet. This repo is the unconditional baseline (V0). A scaled-up, sequence-conditioned version is planned (see Roadmap).

## Pipeline
 
```
PDB IDs → Cα coordinates → pairwise distance matrix → clip + normalise → crop/pad to 64×64
        → UNet DDPM (noise prediction, masked loss) → DDPM sampling → rescale to Å → MDS → 3D Cα trace
```
 
1. **Preprocessing**: downloads structures with Biopython, takes the first chain, extracts one Cα per standard residue, and computes the L×L Euclidean distance matrix. Distances are clipped at 40 Å and scaled to [-1, 1]. Maps are randomly cropped to 64×64 if longer, or padded if shorter, using a sentinel value of `-2.0` that can't occur in real data.
2. **Model**: a UNet with residual blocks, self-attention, and sinusoidal time embeddings, adapted from a from-scratch Stable Diffusion implementation (**[https://github.com/hkproj/pytorch-stable-diffusion]**). Input and output are changed to a single channel. Cross-attention takes a zero dummy context, since this baseline is unconditional.
3. **Training**: noise-prediction objective. The loss is a **masked MSE** computed only over real (non-padded) entries, so the constant padding region doesn't dilute the loss or make it look better than it is.
4. **Sampling** (`ddpm.py`, `sample.py`): DDPM ancestral sampling with 50 inference steps. The predicted x0 is clamped to [-1, 1] at each step. Output is rescaled to 0–40 Å.
5. **Reconstruction**: the generated map is symmetrised, and embedded in 3D with metric MDS.


## What works
 
- Data preprocessing from PDB IDs to fixed-size, normalised maps (10 structures covering different fold classes)
- Training loop with a padding-aware loss
- DDPM noise schedule, forward noising and reverse sampling
- Distance map → 3D Cα trace via MDS (verified on real maps from the dataset)

## What doesn't work yet (known issues)
 
- **Samples are poor.** The baseline was trained on only 10 structures (8 train / 2 test) for about 160 gradient steps at batch size 1, which is far too little.
- **No length conditioning.** The model always generates a full 64×64 map, and nothing tells it the protein length, so visualisation crops `[:L, :L]` manually.
- **Generated maps aren't guaranteed to be valid distance matrices** so MDS traces from samples are noisy.
- **Oversized model for the data.** The UNet has ~37M parameters, and attention at the 64×64 level is expensive and unnecessary for a dataset this small.

## Usage
 
```bash
pip install torch numpy biopython scikit-learn matplotlib tqdm
 
# then run the training and sampling notebook: [unconditional.ipynb]
```
 
## Roadmap
 
- [ ] Scale to a clustered, non-redundant set of a few thousand protein domains, split by cluster
- [ ] Apply random crops at load time (data augmentation) instead of once in preprocessing
- [ ] Smaller UNet, with attention only at lower resolutions
- [ ] Train for tens of thousands of steps with batch size > 1 and a lower learning rate
- [ ] Condition on ESM-2 sequence embeddings (precomputed and cached)
- [ ] Evaluate with proper metrics (symmetry error, contact precision, nearest-training-map distance)

## Acknowledgements
 
- UNet / attention / DDPM structure adapted from **[(https://github.com/hkproj/pytorch-stable-diffusion)]**
- DDPM: Ho et al., 2020, https://arxiv.org/abs/2006.11239
- Structures from the RCSB Protein Data Bank

