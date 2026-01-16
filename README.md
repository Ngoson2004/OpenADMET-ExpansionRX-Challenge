# ExpansionRX-OpenADMET challenge

## 1. Chemprop GNN
Graph Neural Network is the main deep learning solution that we use for our work. We use a library called Chemprop to build, configure and train this model for ADME prediction. The model is trained on 5 different random splits, with checkpoint of each split saved. Submission is made by averaging predictions from 5 model checkpoints.

## 2. Method
### 2.1 Data curation
We perform data transformation using logarithm base 10 on all endpoints of ExpansionRX training dataset, except logD and MGMB. We implement logit percentage base 10 for MGMB, while LogD values remain the same.

In addition, we increase training data size by fetching other datasets: 
 - Galapagos' curated dataset for Polaris challenge [1]
 - 300k molecules Novartis' dataset [2][3] (create by merging a 16k molecules and a 274k molecules set)

These helper datasets went through chemical curation - where invalid/unusual molecules are filtered out and SMILES are canonicalised. For endpoints that are not normally distributed, they are transformed with logarithm base 10. All curation operations are carried out using the Bart Lenselink's tool, published on GitHub [4]

### 2.2 Molecular featurisation
For featurisation of molecules, we mainly used graph-based representation (MolGraph). Additionally, we experimented with combining Molgraph with other auxiliary featurisers: CDD [5] and Maplight [6] features.

CDD features are generate by CDD Vault - a cheminformatics software for drug discovery. It calculates a total of 17 additional properties for each molecules. However, we only select the followings as helper tasks for training:
* **logP**: intrinsic lipophilicity; log of the octanol/water partition coefficient (P).
* **logD**: lipophilicity at a given pH; logP adjusted for ionization (charged vs neutral forms).
* **logS**: log solubility in water (amount dissolved per mL/L, depending on dataset units).
* **pKa**: pH where 50% protonated / 50% deprotonated (may refer to the pKa nearest physiological pH ~7.4).
* **pKa (Acidic)**: lowest pKa value (most acidic site).
* **pKa (Basic)**: highest pKa value (most basic site).  

Maplight is a concatenation of the following featurisers:
* **Morgan fingerprint (ECFP)**: circular, hashed substructure fingerprint capturing atom neighborhoods up to a chosen radius.
* **Avalon fingerprint**: hashed structural fingerprint (path/substructure-based).
* **Extended-Reduced Graph (ErG) fingerprint**: pharmacophore-style fingerprint encoding reduced graph features (e.g., H-bonding, charge, hydrophobics) and their relationships.
* **MACCS keys**: fixed-length fingerprint of predefined structural fragments (“keys”).

### 2.3 Multitask learning
Multitask learning is a training method which allows a model to learn more than one target simultaneously. Conventional model fine-tuning includes input data (labelled as X) and ONE true value data (labelled as Y). With multitask learning, however, a model can be trained on multiple true value data. In other words, there is more than one Y.

In our work, multitask learning is implemented intensively to boost the performance of GNN.

## 3. Hyperparameters optimisation
Decision on which hyperparameter values to configure is also important. We realise this has significant effect on model performance along with the training methods. We tested automatic hyperparameter optimisation with Raytune's Tree-structured Parzen Estimators algorithm, then compared it with manual hyperparameters picking.

Eventually, manual decision yeilds better performance than automatic optimisation. The hyperparameters we chose are listed here:

| Hyperparameter       | Value            |
|----------------------|------------------|
| depth                | 6                |
| ffn_hidden_dim       | 512              |
| ffn_num_layers       | 2                |
| message_hidden_dim   | 2048             |
| dropout              | 0.1              |
| init_lr              | 0.000001         |
| max_lr               | 0.001            |
| final_lr             | 0.0001           |
| warmup               | 5                |
| batch_size           | 256              |
| weight_decay         | 0.0001           |

## 4. Model performance during training
We evaluate model based on 5 validation sets from random splits, then average the MAE calculated on each set. MAE metrics are estimated separately for each ADMET endpoints.

Results show that utilising CDD featuriser, proper data curation, increasing training data by fetching molecules and endpoints from other datasets, along with appropriate selection of hyper-parameters create positive impact on model's performance.

## Reference
[1] K. Goossens, G. Tricarico, Johan Hofmans, Marie-Pierre Dréanic, S. de Cesco, and Eelke Bart Lenselink, “ChemProp multi-task models for predicting ADME properties in the Polaris challenge,” ChemRxiv, Jun. 2025, doi: https://doi.org/10.26434/chemrxiv-2025-q12vh.

[2] A. Fluetsch, M. Trunzer, G. Gerebtzoff, and R. Rodríguez-Pérez, “Deep Learning Models Compared to Experimental Variability for the Prediction of CYP3A4 Time-Dependent Inhibition,” Chemical Research in Toxicology, vol. 37, no. 4, pp. 549–560, Mar. 2024, doi: https://doi.org/10.1021/acs.chemrestox.3c00305.

[3] G. Peteani, M. T. D. Huynh, G. Gerebtzoff, and R. Rodríguez-Pérez, “Application of machine learning models for property prediction to targeted protein degraders,” Nature Communications, vol. 15, no. 1, Jul. 2024, doi: https://doi.org/10.1038/s41467-024-49979-3.

[4] lenselinkbart, “GitHub - lenselinkbart/Datautils,” GitHub, 2025. https://github.com/lenselinkbart/Datautils (accessed Jan. 14, 2026).

[5] Data were archived and analyzed using the CDD Vault from Collaborative Drug Discovery (Burlingame, CA; www.collaborativedrug.com)

[6] J. H. Notwell and M. W. Wood, “ADMET property prediction through combinations of molecular fingerprints,” arXiv (Cornell University), Sep. 2023, doi: https://doi.org/10.48550/arxiv.2310.00174.