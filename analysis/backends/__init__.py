"""Model backends for the CORIVA runner. Heavy deps (torch/transformers) are
imported lazily inside each backend so the runner loads without a GPU stack."""
