from Bio import SeqIO
from tqdm import tqdm


def extract_chunks_from_fasta(fasta_file, chunk_size=1000, stride=500):
    """
    Extract fixed-length DNA chunks from sequences in a FASTA file.
    """
    chunks = []
    with open(fasta_file, "r") as handle:
        for record in tqdm(SeqIO.parse(handle, "fasta"), desc="Reading FASTA"):
            seq = str(record.seq).upper()
            for i in range(0, len(seq) - chunk_size + 1, stride):
                chunk = seq[i:i + chunk_size]
                if "N" not in chunk:  # Skip low-quality regions
                    chunks.append(chunk)
    return chunks


def build_kmer_vocab(k):
    """
    Build a vocabulary of all possible k-mers.
    """
    from itertools import product
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    vocab = {kmer: idx for idx, kmer in enumerate(kmers)}
    return vocab


def tokenize_chunks(chunks, k, vocab):
    """
    Tokenize DNA chunks into overlapping k-mer indices.
    """
    tokenized = []
    unk_idx = len(vocab)  # For any unknown k-mers (should be rare or filtered)
    for chunk in tqdm(chunks, desc="Tokenizing"):
        tokens = []
        for i in range(len(chunk) - k + 1):
            kmer = chunk[i:i + k]
            idx = vocab.get(kmer, unk_idx)
            tokens.append(idx)
        tokenized.append(tokens)
    return tokenized
