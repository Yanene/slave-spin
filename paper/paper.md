---
title: 'pySSMF: A Python package for slave spin mean field method'
tags:
  - Python
  - Condensed matter Physics
  - Strongly correlated electron systems
  - Hubbard model
  - slave spin mean field method
authors:
  - name: Youssra. Anene
    #orcid: 0000-0000-0000-0000
    corresponding: true 
    #equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: José M. Pizarro
    orcid: 'https://orcid.org/0000-0002-6751-8192'
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Lorenzo. Fratino
    orcid: 'https://orcid.org/0000-0003-1288-2859'
    #corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: LPTM, CY Cergy Paris University, France
   index: 1
   #ror: 00hx57361
 - name: 'Federal Institute for Materials Research and Testing (BAM), Germany'
   index: 2
date: 21 May 2026
bibliography: paper.bib
#csl: ieee.csl

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
....

# Statement of need

Quantum materials such as high-temperature superconductors, Mott insulators, and heavy-fermion compounds exhibit electronic behaviors that conventional single-particle frameworks cannot adequately capture. The standard tight-binding approximation, though computationally convenient, neglects the electron-electron correlations that govern the physics of these systems. Obtaining renormalized band structures that faithfully encode many-body effects is therefore a fundamental requirement for realistic modeling. While methods like DFT+DMFT have proven powerful in this regard, their computational cost makes them poorly suited for high-throughput screening or broad exploratory studies across parameter space.
Slave spin mean field (SSMF) theory offers a compelling middle ground: it captures essential correlation-driven phenomena, including Mott transitions and quasiparticle weight renormalization,  at a fraction of the cost of DMFT. Despite this practical appeal, the community currently lacks an open-source, standardized implementation of SSMF. Existing codes are largely confined to individual research groups, are rarely documented for general use, and are not structured to support systematic or reproducible calculations. This absence creates real barriers to validation, benchmarking, and methodological extension.
`pySSMF` addresses this gap directly. It is a Python package that implements SSMF theory for the two-dimensional Hubbard model, supporting both single- and multi-orbital cases across paramagnetic and antiferromagnetic phases and a range of lattice geometries. The package is designed with efficient parameter scanning in mind, enabling rapid computation of quasiparticle weights across large regions of model space. By making this methodology openly available with a clean interface and reproducible workflows, SSMF lowers the barrier to entry for condensed matter physicists and adjacent researchers seeking to study strongly correlated electron systems without the overhead of more demanding many-body frameworks.

# State of the field    

The study of strongly correlated electron systems is shaped by a persistent tension between accuracy and computational cost. While DMFT and its extensions, cluster DMFT, GW+EDMFT,  describe multiorbital Hubbard physics with high fidelity, their expense makes broad parameter exploration impractical for many groups. Recent progress in downfolding, low-energy effective models, and machine-learned mappings from exact diagonalization has shown that semiquantitative accuracy is achievable at substantially lower cost, pointing toward an emerging generation of lightweight, physically grounded correlation-aware solvers. SSMF theory sits naturally within this landscape. By mapping the electron occupation onto auxiliary spin degrees of freedom and treating interactions at the mean-field level, SSMF reproduces key signatures of correlated phases, including Mott transitions, and antiferromagnetic phases, while remaining computationally inexpensive enough for systematic studies. Benchmarks against DMFT have shown good agreement for quasiparticle weights and phase boundaries across a range of filling and interaction strengths, validating its use as a practical tool for exploring multiorbital two-dimensional Hubbard models in both paramagnetic and antiferromagnetic regimes.
At the same time, the broader field of computational condensed matter physics has undergone a significant cultural shift toward open, reproducible software practices. Projects such as TRIQS have demonstrated the scientific value of modular, community-maintained frameworks: they lower barriers to entry, facilitate benchmarking across independent implementations, and extend the usable lifetime of scientific codes well beyond individual research projects. Despite this trend, no comparable open-source implementation of SSMF theory currently exists. Existing codes are overwhelmingly private, undocumented for external use, or tightly coupled to the workflows of specific groups, conditions that hinder reproducibility and make cross-study comparisons unreliable.
`pySSMF` is designed to address this absence directly. By providing a documented, tested, and extensible Python implementation, complete with benchmark datasets, Jupyter notebook tutorials, and automated test suites, it brings slave-spin methodology into alignment with contemporary standards for open scientific software. The package is intended to serve both experienced researchers seeking a reliable tool for systematic exploration and students entering the field who need a well-structured codebase to learn from and build upon.

# Software design

...

# Research impact statement

Beyond its immediate utility, `pySSMF` is built to grow. The current implementation already goes further than most comparable tools by incorporating magnetic phases, including antiferromagnetic order, alongside the standard paramagnetic case, which meaningfully widens the range of physics one can study. Support for real materials beyond model Hamiltonians is a natural next step, and we plan to include this in a future release.
The longer-term vision is more ambitious. A fast, systematic tool like this one is well-suited to high-throughput exploration: running calculations across large swaths of parameter or chemical space becomes practical in a way it simply isn't with DMFT. Done at scale, this could produce datasets of renormalized bands, quasiparticle weights, and effective masses that are otherwise hard to assemble cheaply. Those datasets, in turn, are exactly the kind of structured, physics-rich data that machine learning models, graph neural networks, transformers, can learn from effectively, potentially uncovering relationships between crystal structure, chemical composition, and correlated electronic behavior that are not obvious from theory alone. We believe this positions `pySSMF` as a useful building block in a broader computational pipeline aimed at accelerating the discovery of materials with targeted properties, whether that means unconventional superconductivity, large thermoelectric response, or switching behavior relevant to neuromorphic computing.

# Mathematics

The central model is the multi-orbital Hubbard Hamiltonian,

$$
H =
\sum_{ij,m,\sigma} t_{ij}
\left(
d^{\dagger}_{im\sigma} d_{jm\sigma}
+ \mathrm{h.c.}
\right)
- \mu \sum_{i,m,\sigma} n_{im\sigma}
+ U \sum_{i,m} n_{im\uparrow} n_{im\downarrow}
+ U' \sum_{i,m \neq m'} n_{im\uparrow} n_{im'\downarrow}
+ (U' - J)
\sum_{i,m < m',\sigma}
n_{im\sigma} n_{im'\sigma}.
$$

where $$d^{\dagger}_{im\sigma}$$ ($$d_{im\sigma}$$) creates (annihilates) an electron at site $$i$$,
orbital $$m$$, spin $$\sigma$$, and $$n_{im\sigma} = d^{\dagger}_{im\sigma} d_{im\sigma}$$.
The chemical potential $$\mu$$ controls the filling, $$U$$ and $$U'$$ are the intra- and
inter-orbital Coulomb repulsions, and $$J$$ is Hund's coupling. We adopt the
rotationally invariant relation $$U' = U - 2J$$ throughout. Here it is with zero crystal field splitting. 

## Slave-spin decomposition

The $$Z_2$$ slave-spin method following these papers @de2005orbital and @de2017modeling enlarges the Hilbert
space by representing each physical fermion as a product of an auxiliary fermion
$$f_{im\sigma}$$ and a slave-spin operator $$O_{im\sigma}$$,

$$
d_{im\sigma} \rightarrow f_{im\sigma}\, O_{im\sigma}, \qquad
d^{\dagger}_{im\sigma} \rightarrow O^{\dagger}_{im\sigma} f^{\dagger}_{im\sigma},
$$

with $$O_{im\sigma} = S^-_{im\sigma} + c_{im\sigma} S^+_{im\sigma}$$, where
$$S^{\alpha}_{im\sigma}$$ are Pauli operators and the gauge parameter $$c_{im\sigma}$$
is fixed by the local occupancy. The physical subspace is recovered by enforcing

$$
f^{\dagger}_{im\sigma} f_{im\sigma} = S^z_{im\sigma} + \tfrac{1}{2},
$$

via site-dependent Lagrange multipliers $$\lambda_{im\sigma}$$. Under this mapping,
all density–density interactions reduce to products of $S^z$ operators, and the
Hamiltonian separates into two coupled sectors: a free-fermion part renormalized
by slave spin expectation values, and a local interacting spin problem. Next, we aim to decouple the fermions from the spins that's why we apply a mean field decoupling between the two sectors that yields a self consistent equations: The fermionic sector reduces to a renormalized tight-binding problem, from which
the quasiparticle weight

$$
Z_{m\sigma} = \langle O_{im\sigma} \rangle^2
$$

is extracted. The slave spin sector reduces to a single-site Ising-like
Hamiltonian solved exactly at each iteration. Self-consistency is achieved when
the constarint condition satisfied.
To access symmetry-broken phases beyond the paramagnetic solution, we follow the
variational extension of @crispino2023slave. Here additional parameters appear like $$\lambda_{im\sigma}^0$$ which is zero when we are in the paramgnetic regime and we introduce the notion of the sublattices and we impose symmetry breaking with imposing that sublattice A is spin up and sublattice B is spin down and so we have additional indix for the sublattice for the parameters.

The self-consistent loop mentioned earlier is the computational core of `pySSMF`. The package
iterates the fermionic and slave spin sectors for arbitrary orbital number,
filling, and lattice geometry, returning converged quasiparticle weights and
renormalized band structures. Dedicated routines handle antiferromagnetic
order parameters, enabling the magnetic phase boundaries discussed in the
results section.


# Figures

...
# AI usage disclosure

AI tools, including Codex, were used in a limited way to support brainstorming of software design ideas. 
No AI tools were used for implementation, data analysis, or scientific interpretation. The code and results were fully developed, tested, and validated by the authors.
 

# References
