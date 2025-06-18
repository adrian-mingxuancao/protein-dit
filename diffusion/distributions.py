import torch

class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes (residues) in the dataset, and sample from this distribution.
            histogram: dict. The keys are num_nodes, the values are counts
        """
        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)
        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p


class DistributionProteinSequence:
    def __init__(self, sequence_lengths):
        """ Compute the distribution of protein sequence lengths in the dataset.
            sequence_lengths: dict. The keys are sequence lengths, the values are counts
        """
        if type(sequence_lengths) == dict:
            max_length = max(sequence_lengths.keys())
            prob = torch.zeros(max_length + 1)
            for length, count in sequence_lengths.items():
                prob[length] = count
        else:
            prob = sequence_lengths

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_length(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_lengths):
        assert len(batch_lengths.size()) == 1
        p = self.prob.to(batch_lengths.device)
        probas = p[batch_lengths]
        log_p = torch.log(probas + 1e-30)
        return log_p


class DistributionAminoAcids:
    def __init__(self, amino_acid_counts):
        """ Compute the distribution of amino acid types in the dataset.
            amino_acid_counts: dict. The keys are amino acid types, the values are counts
        """
        if type(amino_acid_counts) == dict:
            total = sum(amino_acid_counts.values())
            prob = torch.zeros(len(amino_acid_counts))
            for i, (aa, count) in enumerate(sorted(amino_acid_counts.items())):
                prob[i] = count / total
        else:
            prob = amino_acid_counts

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_aa(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_aas):
        assert len(batch_aas.size()) == 1
        p = self.prob.to(batch_aas.device)
        probas = p[batch_aas]
        log_p = torch.log(probas + 1e-30)
        return log_p 