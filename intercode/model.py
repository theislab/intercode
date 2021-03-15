import numpy as np
import torch
import pickle
import os
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
    activs_key
        Key for `adata.varm` where the activities binary array is stored.
    terms_key
        Key for `adata.uns` where the terms' names are stored.
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
    def __init__(
        self,
        adata,
        activs_key = 'I',
        terms_key = 'terms',
        n_sparse = None,
        n_dense = None,
        use_cuda = True,
        **kwargs
    ):
        if 'terms' not in adata.uns or 'I' not in adata.varm:
            raise ValueError("The AnnData object must have annotations.")

        self.adata = adata

        n_ann = len(adata.uns['terms'])
        n_vars = self.adata.n_vars
        self.model = AutoencoderLinearDecoder(n_vars, n_ann, n_sparse, n_dense, **kwargs)

        if use_cuda and torch.cuda.is_available():
            self.model.cuda()

        self._annotation_params = {}
        self._annotation_params['var_names'] = list(adata.var_names)
        self._annotation_params['term_names'] = list(adata.uns[terms_key])
        self._annotation_params['activities'] = adata.varm[activs_key].tolist()

        self._init_params = {}
        self._init_params['n_sparse'] = n_sparse
        self._init_params['n_dense'] = n_dense
        self._init_params['use_cuda'] = use_cuda
        self._init_params['activs_key'] = activs_key
        self._init_params['terms_key'] = terms_key
        self._init_params.update(kwargs)


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
                          test_data, optim, self._init_params['activs_key'], **kwargs)

    def encode(self, x, term_names=None):
        """\
        Get latent representation of the input.

        Latent dimension coorespond to gene sets in the adata used for tarining.
        The names and order of gene sets can be found in `adata.uns`.

        Parameters
        ----------
        x
            Data to be transformed.
        term_names
            Return only the latent dimensions corresponding to these
            annotations' / terms' names.

        Returns
        ----------
        Latent representation of `x`.
        """
        x = torch.as_tensor(x)
        using_cuda = next(self.model.parameters()).is_cuda
        if using_cuda:
            x = x.cuda()

        encoded = self.model.encoder(x).detach().cpu().numpy()

        if term_names is not None:
            idx = [self.term_names.index(name) for name in term_names]
            return encoded[:, idx]
        else:
            return encoded


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
        for k, v in self.model.decoder.weight_dict.items():
            d[k] = (v.data.norm(p=2, dim=0)>0).cpu().numpy()
        return d

    @property
    def term_names(self):
        return self._annotation_params['term_names']

    @property
    def var_names(self):
        return self._annotation_params['var_names']

    def save(self, dir_path, overwrite=False):
        """\
        Save the state of the model.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        """
        model_path = os.path.join(dir_path, "model_params.pt")
        annot_params_path = os.path.join(dir_path, "annot_params.pkl")
        init_params_path = os.path.join(dir_path, "init_params.pkl")

        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)

        torch.save(self.model.state_dict(), model_path)
        with open(annot_params_path, "wb") as f:
            pickle.dump(self._annotation_params, f)
        with open(init_params_path, "wb") as f:
            pickle.dump(self._init_params, f)

    @classmethod
    def load(cls, dir_path, adata):
        """\
        Instantiate a model from the saved output.

        Parameters
        ----------
        dir_path
            Path to saved outputs.
        adata
            AnnData object with annotations.

        Returns
        ----------
            Model with loaded state dictionaries.
        """
        model_path = os.path.join(dir_path, "model_params.pt")
        annot_params_path = os.path.join(dir_path, "annot_params.pkl")
        init_params_path = os.path.join(dir_path, "init_params.pkl")

        model_state_dict = torch.load(model_path)

        with open(annot_params_path, "rb") as handle:
            annot_params = pickle.load(handle)

        if len(annot_params['var_names']) != adata.n_vars:
            raise ValueError('n_vars in the adata doesn\'t match n_vars of the model')

        with open(init_params_path, "rb") as handle:
            init_params = pickle.load(handle)

        print('Inserting terms\' names and activities to the Anndata object.')
        adata.uns[init_params['terms_key']] = annot_params['term_names']
        adata.varm[init_params['activs_key']] = np.array(annot_params['activities'])

        new_intercode = cls(adata, **init_params)
        new_intercode.model.load_state_dict(model_state_dict)
        new_intercode.model.eval()

        return new_intercode
