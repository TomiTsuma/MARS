from model.metrics import *
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles-gen", type=str,
                            default="runs/phase1/samples.smi")
    parser.add_argument("--smiles-train", type=str,
                        default='artifacts/processed/train_smiles.txt')
    args = parser.parse_args()
    smiles_generated = []
    smiles_train = []
    with open(args.smiles_gen, 'rb') as file:
        for line in file.readlines():
            smiles_generated.append(line.strip())
    with open(args.smiles_train, 'rb') as file:
            for line in file.readlines():
                smiles_train.append(line.strip())

    print(smiles_generated)
    print(smiles_train[0:4])
    bm = basic_metrics(set(smiles_generated), set(smiles_train))
    print(f"Basic metrics VUN: {bm}")
    id = internal_diversity(set(smiles_generated))
    print(f"Internal diversity: {id}")
    nn = snn(smiles_generated, smiles_train)
    print(f"Tanimoto similarity: {nn}")
    ss = scaffold_similarity(set(smiles_generated), set(smiles_train))
    print(f"Scaffold similarity: {ss}")
    fcd_dist = fcd(set(smiles_generated), set(smiles_train))
    print(f"Frechet ChemNet Distance: {fcd_dist}")
    ps = property_summary(smiles_generated)
    print(f"Property summary: {ps}")

if __name__ == "__main__":
     main()
