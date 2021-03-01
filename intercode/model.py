import torch
from .linear_decoder import AutoencoderLinearDecoder, train_autoencoder


class Intercode:
    """\
    This class contains the inplementation of interpretable autoencoder.

    Latent dimensions learned by the autoencoder correspond to gene sets
    added to an AnnData object.

    Parameters
    ----------
    adata
        Annotated data matrix. It should contain annotations of genes by gene sets.
    n_sparse
        Number of dimensions corresponding to sparse terms in the latent representation.
        These dimensions don't requre annotations.
    n_dense
        Number of dimensions corresponding to dense terms in the latent representation.
        These dimensions don't requre annotations.
    use_cuda
        Use GPU acceleration.
    **kwargs
        Additional arguments specifying the model architecture.
    """
    def __init__(self, adata, n_sparse=None, n_dense=None, use_cuda=True, **kwargs):
        if 'terms' not in adata.uns or 'I' not in adata.varm:
            raise ValueError("The AnnData object must have annotations.")

        self.adata = adata

        n_ann = len(adata.uns['terms'])
        n_vars = self.adata.n_vars
        self.model = AutoencoderLinearDecoder(n_vars, n_ann, n_sparse, n_dense, **kwargs)

        if use_cuda and torch.cuda.is_available():
            self.model.cuda()

    def train(self, lr, batch_size, num_epochs,
              l2_reg_lambda0=0.1, lambda1=None, lambda2=None, lambda3=None,
              test_data=None, optim=torch.optim.Adam, **kwargs):
        """\
        Train the model.

        Latent dimension coorespond to gene sets in the adata used for tarining.
        The names and order of gene sets can be found in `adata.uns`.

        Parameters
        ----------
        lr
            Learning rate for training the model.
        batch_size
            Defines the batch size that is used during each Iteration.
        num_epochs
            Number of epochs to train the model for.
        l2_reg_lambda0
            L2 regularization coefficient for the encoder output.
        lambda1
            L1 regularization coefficient for inactive genes in gene sets' annotations.
        lambda2
            L1 regularization coefficient for unannotated sparse terms / dimensions.
        lambda3
            Group lasso regularization coefficient.
        test_data
            Data for validation. If not provided, uses the full training dataset.
        optim
            Optimizer to use for training.
        **kwargs
            Additional arguments for the optimizer.
        """
        if lambda1 is not None:
            lambda1 *= lr
        if lambda2 is not None:
            lambda2 *= lr
        if lambda3 is not None:
            lambda3 *= lr
        train_autoencoder(self.adata, self.model, lr, batch_size, num_epochs,
                          l2_reg_lambda0, lambda1, lambda2, lambda3,
                          test_data, optim, **kwargs)

    def encode(self, x):
        """\
        Get latent representation of the input.

        Latent dimension coorespond to gene sets in the adata used for tarining.
        The names and order of gene sets can be found in `adata.uns`.

        Parameters
        ----------
        x
            Data to be transformed.

        Returns
        ----------
        Latent representation of `x`.
        """
        x = torch.as_tensor(x)
        using_cuda = next(self.model.parameters()).is_cuda
        if using_cuda:
            x = x.cuda()
        return self.model.encoder(x)

    def nonzero_terms(self):
        """\
        Get indices for active terms

        Some latent dimensions get deactivated during training due to regularization.
        This method allows to get indices of active terms. Inactive terms / dimensions
        don't have any influence on reconstruction.

        Returns
        ----------
        Indices of active terms.
        """
        d = {}
        for k, v in self.model.decoder.weight_dict:
            d[k] = (v.data.norm(p=2, dim=0)>0).cpu().numpy()
        return d
