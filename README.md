
# DiffAE

This is a fork of [Diffusion Autoencoders](https://github.com/phizaz/diffae.git) pruned for face generation and editing.

## Original Paper
```bibtex
@inproceedings{preechakul2021diffusion,
      title={Diffusion Autoencoders: Toward a Meaningful and Decodable Representation}, 
      author={Preechakul, Konpat and Chatthee, Nattanat and Wizadwongsa, Suttisak and Suwajanakorn, Supasorn},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
      year={2022},
}
```

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Download pretrained models:

- FFHQ256 DiffAE: [Checkpoint](https://drive.google.com/drive/folders/1-5zfxT6Gl-GjxM7z9ZO2AHlB70tfmF6V)
- FFHQ256 DiffAE with latent DPM: [Checkpoint](https://drive.google.com/drive/folders/1-H8WzKc65dEONN-DQ87TnXc23nTXDTYb)
- FFHQ256 Classifier: [Checkpoint](https://drive.google.com/drive/folders/117Wv7RZs_gumgrCOIhDEWgsNy6BRJorg)

Place checkpoints in `checkpoints` directory.

## License

MIT License
