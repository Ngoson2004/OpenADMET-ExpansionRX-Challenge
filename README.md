# ExpansionRX-OpenADMET challenge

## 1. Chemprop GNN
Graph Neural Network is the main deep learning solution that we use for our work. We use Chemprop to build, configure and train this model for ADME prediction. The model is trained on 5 different random splits, with checkpoint of each split saved. Submission is made by averaging predictions from 5 model checkpoints.

## 2. Method
### 2.1 Multitask learning and datasets

We perform data transformation using logarithm base 10 on all endpoints of the ExpansionRX training dataset, except logD and MGMB. We implement logit percentage base 10 for MGMB, while LogD values remain untransformed.

We apply multitask learning with Chemprop and gather datasets:

- Galapagos' curated dataset for Polaris challenge [1]
- 300K molecules Novartis' dataset [2][3] (created by merging a 16K molecules and a 274K molecules set)

We complement these datasets with calculated values for logD and pKa from CDD [4].

These helper datasets underwent chemical curation, where invalid/unusual molecules were filtered out, and SMILES were canonicalised. For endpoints that are not normally distributed, they are transformed with logarithm base 10. All curation operations are carried out using Datautils by Bart Lenselink [5]

### 2.2 Molecular featurisation
For featurisation of molecules, we mainly used graph-based representation (MolGraph). Additionally, we experimented with combining Molgraph with other auxiliary featurisers: Maplight [6] features, which is a concatenation of the following featurisers:

- Morgan fingerprint (ECFP): circular, hashed substructure fingerprint capturing atom neighborhoods up to a chosen radius.
- Avalon fingerprint: hashed structural fingerprint (path/substructure-based).
- Extended-Reduced Graph (ErG) fingerprint: pharmacophore-style fingerprint encoding reduced graph features (e.g., H-bonding, charge, hydrophobics) and their relationships.
- MACCS keys: fixed-length fingerprint of predefined structural fragments (“keys”).

## 3. Hyperparameter optimisation
We tested automatic hyperparameter optimisation with Raytune's Tree-structured Parzen Estimators algorithm, then compared it with manual hyperparameters picking.

Eventually, manual decision yields better performance. The hyperparameters we chose are listed here:

| Hyperparameter      | Value    |
|---------------------|---------:|
| depth               | 6        |
| ffn_hidden_dim      | 512      |
| ffn_num_layers      | 2        |
| message_hidden_dim  | 2048     |
| dropout             | 0.1      |
| init_lr             | 0.000001 |
| max_lr              | 0.001    |
| final_lr            | 0.0001   |
| warmup              | 5        |
| batch_size          | 256      |
| weight_decay        | 0.0001   |

## 4. Model performance during training
We evaluate models based on 5 validation sets from random splits, then average the MAE calculated on each set. MAE metrics are estimated separately for each ADMET endpoints.

Our most drastic performance improvement was achieved through expert, manual hyperparameter optimisation on ExpansionRx data only, which enabled quick model train-test cycles and fast iteration. We then ported these hyperparameters to multitask and additional featurisation, but due to lack of time, didn’t explore once again hyperparameter tuning which we expect might help.

Careful data curation, additional public ADME data provided additional performance improvements over the tuned chemprop model trained only on ExpansionRx data.

## Reference
[1] K. Goossens, G. Tricarico, Johan Hofmans, Marie-Pierre Dréanic, S. de Cesco, and Eelke Bart Lenselink, “ChemProp multi-task models for predicting ADME properties in the Polaris challenge,” ChemRxiv, Jun. 2025, doi: https://doi.org/10.26434/chemrxiv-2025-q12vh.

[2] A. Fluetsch, M. Trunzer, G. Gerebtzoff, and R. Rodríguez-Pérez, “Deep Learning Models Compared to Experimental Variability for the Prediction of CYP3A4 Time-Dependent Inhibition,” Chemical Research in Toxicology, vol. 37, no. 4, pp. 549–560, Mar. 2024, doi: https://doi.org/10.1021/acs.chemrestox.3c00305.

[3] G. Peteani, M. T. D. Huynh, G. Gerebtzoff, and R. Rodríguez-Pérez, “Application of machine learning models for property prediction to targeted protein degraders,” Nature Communications, vol. 15, no. 1, Jul. 2024, doi: https://doi.org/10.1038/s41467-024-49979-3.

[4] Data were archived and analyzed using the CDD Vault from Collaborative Drug Discovery (Burlingame, CA; www.collaborativedrug.com)

[5] lenselinkbart, “GitHub - lenselinkbart/Datautils,” GitHub, 2025. https://github.com/lenselinkbart/Datautils (accessed Jan. 14, 2026).

[6] J. H. Notwell and M. W. Wood, “ADMET property prediction through combinations of molecular fingerprints,” arXiv (Cornell University), Sep. 2023, doi: https://doi.org/10.48550/arxiv.2310.00174.