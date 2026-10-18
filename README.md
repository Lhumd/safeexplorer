# SafeExplorer project page

Project page for the TMLR 2026 paper *SafeExplorer: An Unbiased Mixed-Policy Gradient for Reinforcement Learning with Recovery Interventions* by Elham Daneshmand, Majid Khadiv, Glen Berseth and Hsiu-Chin Lin (McGill University, Technical University of Munich, Université de Montréal, Mila).

- Page: https://lhumd.github.io/safeexplorer/
- Paper: https://openreview.net/forum?id=ppOrbSsMM0 (PDF also at https://lhumd.github.io/safeexplorer/paper.pdf)

## Citation

```bibtex
@article{daneshmand2026safeexplorer,
  title   = {{SafeExplorer}: An Unbiased Mixed-Policy Gradient for Reinforcement Learning with Recovery Interventions},
  author  = {Daneshmand, Elham and Khadiv, Majid and Berseth, Glen and Lin, Hsiu-Chin},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=ppOrbSsMM0}
}
```

GitHub's "Cite this repository" button reads `CITATION.cff`.

## Contents

```
index.html     the page (static, math pre-rendered with KaTeX, works without JavaScript)
index.md       the same content as Markdown with TeX equations
llms.txt       short summary with links for AI assistants (llmstxt.org)
paper.pdf      camera-ready paper
static/        styles, a small script, figures, rollout clips, KaTeX stylesheet and fonts
```

To preview locally, run `python3 -m http.server 8000` in this folder and open http://localhost:8000/.

## Licenses

The paper, its text and its figures are licensed under CC BY 4.0, as are all TMLR papers; the authors keep the copyright. KaTeX in `static/vendor/katex/` is MIT licensed, see `static/vendor/katex/LICENSE`. Inter, Source Serif 4 and JetBrains Mono load from Google Fonts under the SIL Open Font License.

Contact: elham.daneshmand@mail.mcgill.ca
