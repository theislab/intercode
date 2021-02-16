import torch
from .linear_decoder import AutoencoderLinearDecoder, train_autoencoder


class Intercode:

    def __init__(self, adata, n_sparse=None, n_dense=None, use_cuda=True, **kwargs):
        if 'terms' not in adata.uns or 'I' not in adata.varm:
            raise ValueError("The AnnData object must have annotations.")

        self.adata = adata

        n_ann = len(adata.uns['terms'])
        n_vars = self.adata.n_vars
        self.model = AutoencoderLinearDecoder(n_vars, n_ann, n_sparse, n_dense, **kwargs)

        if use_cuda and torch.cuda.is_available():
            self.model.cuda()

    def train(lr, batch_size, num_epochs,
              l2_reg_lambda0=0.1, lambda1=None, lambda2=None, lambda3=None,
              test_data=None, optim=torch.optim.Adam, **kwargs):
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
        x = torch.as_tensor(x)
        using_cuda = next(self.model.parameters()).is_cuda
        if using_cuda:
            x = x.cuda()
        return self.model.encoder(x)

    def nonzero_terms(self):
        d = {}
        for k, v in self.model.decoder.weight_dict:
            d[k] = (v.data.norm(p=2, dim=0)>0).cpu().numpy()
        return d
