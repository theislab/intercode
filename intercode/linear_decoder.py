import torch
import torch.nn as nn
import numpy as np

TERMS_KEYS = ('annotated', 'sparse', 'dense')

# Proximal operators
class ProxOperGroupL2:
    def __init__(self, alpha, omega=None, inplace=True):
    # omega - vector of coefficients with size
    # equal to the number of groups
        if omega is None:
            self._group_coeff = alpha
        else:
            self._group_coeff = (omega*alpha).view(-1)

        self._inplace = inplace

    def __call__(self, W):
        if not self._inplace:
            W = W.clone()

        norm_vect = W.norm(p=2, dim=0)
        norm_g_gr_vect = norm_vect>self._group_coeff

        scaled_norm_vector = norm_vect/self._group_coeff
        scaled_norm_vector+=(~(scaled_norm_vector>0)).float()

        W-=W/scaled_norm_vector
        W*=norm_g_gr_vect.float()

        return W

class ProxOperL1:
    def __init__(self, alpha, I=None, inplace=True):
        self._I = ~I.bool() if I is not None else None
        self._alpha=alpha
        self._inplace=inplace

    def __call__(self, W):
        if not self._inplace:
            W = W.clone()

        W_geq_alpha = W>=self._alpha
        W_leq_neg_alpha = W<=-self._alpha
        W_cond_joint = ~W_geq_alpha&~W_leq_neg_alpha

        if self._I is not None:
            W_geq_alpha &= self._I
            W_leq_neg_alpha &= self._I
            W_cond_joint &= self._I

        W -= W_geq_alpha.float()*self._alpha
        W += W_leq_neg_alpha.float()*self._alpha
        W -= W_cond_joint.float()*W

        return W

# Autoencoder with regularized linear decoder
class CompositeLinearDecoder(nn.Module):
    def __init__(self, n_vars, n_ann=None, n_sparse=None, n_dense=None):
        super().__init__()

        sizes = (n_ann, n_sparse, n_dense)

        if sizes == (None, None, None):
            raise ValueError('At least one type of terms should be chosen')

        self.weight_dict = nn.ParameterDict({})

        N = sum(filter(None, sizes))**0.5
        for i, k in enumerate(TERMS_KEYS):
            if sizes[i] is not None:
                p = nn.Parameter(torch.randn(n_vars, sizes[i]))
                p.data /= N
                self.weight_dict[k] = p

    def forward(self, x):
        vals = tuple(self.weight_dict.values())
        if len(vals) == 1:
            return x.matmul(vals[0].t())
        else:
            return x.matmul(torch.cat(vals, dim=1).t())

    def n_inactive_terms(self):
        n = 0
        for v in self.weight_dict.values():
            n+=(~(v.data.norm(p=2, dim=0)>0)).float().sum().cpu().numpy()

        return int(n)

class AutoencoderLinearDecoder(nn.Module):
    def __init__(self, n_vars, n_ann=None, n_sparse=None, n_dense=None, **kwargs):
        super().__init__()

        sizes = [s if s is not None else 0 for s in (n_ann, n_sparse, n_dense)]

        self.n_vars = n_vars

        self.n_terms = sum(sizes)
        self.n_ann, self.n_sparse, self.n_dense = sizes

        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.mid_layers_size = kwargs.get('mid_layers_size', 400)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_vars, self.mid_layers_size),
            nn.BatchNorm1d(self.mid_layers_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mid_layers_size, self.mid_layers_size),
            nn.BatchNorm1d(self.mid_layers_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mid_layers_size, self.n_terms)
        )

        sizes = [s if s != 0 else None for s in (n_ann, n_sparse, n_dense)]

        self.decoder = CompositeLinearDecoder(n_vars, *sizes)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

def get_prox_operators(I, lambda1, lambda2, lambda3):
    ops_dct = {}

    if lambda3 is not None:
        p_gr = ProxOperGroupL2(lambda3)
    else:
        p_gr = lambda W: W

    if lambda1 is not None:
        p_l1_annot = ProxOperL1(lambda1, I)
        ops_dct[TERMS_KEYS[0]] = lambda W: p_gr(p_l1_annot(W))

    if lambda2 is not None:
        p_l1_sparse = ProxOperL1(lambda2)
        ops_dct[TERMS_KEYS[1]] = lambda W: p_gr(p_l1_sparse(W))

    ops_dct[TERMS_KEYS[2]] = lambda W: p_gr(W)

    return ops_dct

def index_iter(n_obs, batch_size):
    indices = np.random.permutation(n_obs)
    for i in range(0, n_obs, batch_size):
        yield indices[i: min(i + batch_size, n_obs)]

def train_autoencoder(adata, autoencoder, lr, batch_size, num_epochs,
                      l2_reg_lambda0=0.1, lambda1=None, lambda2=None, lambda3=None,
                      test_data=None, optim = torch.optim.Adam, **kwargs):

    optimizer = optim(autoencoder.parameters(), lr=lr, **kwargs)

    I = adata.varm['I']

    n_inact_genes = (1-I).sum()

    use_cuda = next(autoencoder.parameters()).is_cuda

    I = torch.from_numpy(I)

    if use_cuda:
        I = I.cuda()

    prox_ops = get_prox_operators(I, lambda1, lambda2, lambda3)

    term_keys = autoencoder.decoder.weight_dict.keys()
    for k in term_keys:
        if k not in prox_ops:
            raise ValueError('Provide regularization coefficient for '+k)

    if test_data is None:
        t_X = torch.from_numpy(adata.X)
        comment = '-- total train loss: '
        warning_cuda = 'Sending the whole dataset to cuda'
    else:
        t_X = test_data
        comment = '-- test loss:'
        warning_cuda = 'Sending the validation dataset to cuda'
    if use_cuda:
        print(warning_cuda)
        t_X = t_X.cuda()
    test_n_obs = t_X.shape[0]

    l2_loss = nn.MSELoss(reduction='sum')

    if adata.isbacked:
        select_X = lambda adata, selection: torch.from_numpy(adata[selection].X)
    else:
        select_X = lambda adata, selection: torch.from_numpy(adata.X[selection])

    for epoch in range(num_epochs):
        autoencoder.train()

        for step, selection in enumerate(index_iter(adata.n_obs, batch_size)):

            X = select_X(adata, selection)
            if use_cuda:
                X = X.cuda()

            encoded, decoded = autoencoder(X)

            loss = (l2_loss(decoded, X)+l2_reg_lambda0*encoded.pow(2).sum())/len(selection)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in term_keys:
                prox_ops[k](autoencoder.decoder.weight_dict[k].data)

            if step % 100 == 0:
                print('Epoch:', epoch, '| batch train loss: %.4f' % loss.data.cpu().numpy())

        autoencoder.eval()
        t_encoded, t_decoded = autoencoder(t_X)

        t_reconst = l2_loss(t_decoded, t_X).data.cpu().numpy()/test_n_obs
        t_regul = l2_reg_lambda0*t_encoded.pow(2).sum().data.cpu().numpy()/test_n_obs
        t_loss = t_reconst + t_regul

        print('Epoch:', epoch, comment, '%.4f=%.4f+%.4f' % (t_loss, t_reconst, t_regul))

        n_deact_terms = autoencoder.decoder.n_inactive_terms()
        print('Number of deactivated terms:', n_deact_terms)

        n_deact_genes = (~(autoencoder.decoder.weight_dict[TERMS_KEYS[0]].data.abs()>0)&~I.bool()).float().sum()
        print('Share of deactivated inactive genes: %.4f' % (n_deact_genes.cpu().numpy()/n_inact_genes))
