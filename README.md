# -Deep-Learning_multi-layered-perceptron-network
Antigen Discovery for SARS-CoV-2 (“Corona”) Virus Vaccine
Time to end the COVID-19 pandemic by finding potential antigens. The latter are sub-sequences of the virus proteins that can be recognized by our immune system.
Our adaptive immune system consists of 6 HLA (class I) alleles that allow it to selectively identify small fragments of proteins, known as peptides. 
The system is evolved to recognize only peptides of a foreign body and by that invoke an immune proliferation and response of T-cells that destroy the intruder. 
However, unfortunately not all foreign peptides are recognized. For those of you who are interested in learning more about this mechanism.
In this exercise we will train a deep neural network to identify the peptides detected by a specific HLA allele known as HLA_A0201 which is a very common
allele shared by 24% of the western population. The training data consists of ~3,000 positive and ~24,500 negative peptides. Each peptide consists of 9 amino acids
(of 20 types). At a second stage, we will use our trained predictor to identify sequences of 9-amino-acids peptides from the Spike protein of the SARS-CoV-2 virus.
