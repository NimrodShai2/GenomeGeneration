from Bio import SeqIO

records = list(SeqIO.parse("P_Dorei_Genomes/P_Dorei_genome_sequence.fasta", "fasta"))
print(f"Total sequences: {len(records)}")