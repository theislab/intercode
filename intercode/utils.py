import numpy as np
from sklearn.linear_model import LogisticRegression

# add binary I of size n_vars x number of annotated terms in files
# if I[i,j]=1 then gene i is active in annotation j
def add_annotations(adata, files, min_genes=0, max_genes=None, varm_key='I', uns_key='terms'):
    """\
    Add annotations to an AnnData object from files.

    Parameters
    ----------
    adata
        Annotated data matrix.
    files
        Paths to text files with annotations. The function considers rows to be gene sets
        with name of a gene set in the first column followed by names of genes.
    min_genes
        Only include gene sets which have the total number of genes in adata
        greater than this value.
    max_genes
        Only include gene sets which have the total number of genes in adata
        less than this value.
    varm_key
        Store the binary array I of size n_vars x number of annotated terms in files
        in `adata.varm[varm_key]`. if I[i,j]=1 then the gene i is present in the annotation j.
    uns_key
        Sore gene sets' names in `adata.uns[uns_key]`.
    """
    files = [files] if isinstance(files, str) else files
    annot = []

    for file in files:
        with open(file) as f:
            terms = [l.upper().strip('\n').split() for l in f]
        terms = [[term[0].split('_', 1)[-1][:30]]+term[1:] for term in terms if term]
        annot+=terms

    var_names = adata.var_names.str.upper()
    I = [[int(gene in term) for term in annot] for gene in var_names]
    I = np.asarray(I, dtype='int32')


    mask = I.sum(0) > min_genes
    if max_genes is not None:
        mask &= I.sum(0) < max_genes
    I = I[:, mask]
    adata.varm[varm_key] = I
    adata.uns[uns_key] = [term[0] for i, term in enumerate(annot) if i not in np.where(~mask)[0]]


def score_logistic(latents, condition, get_accuracy=False, **kwargs):
    """\
    Train logistic regression and return coefficients or accuracy.

    Parameters
    ----------
    get_accuracy
        Return accuracy (`sklearn.linear_model.LogisticRegression.score`) instead
        of coefficients.
    **kwargs
        Additional arguments for `sklearn.linear_model.LogisticRegression`.

    Returns
    ----------
    Dcitionary of coefficients for each class or accuracy.
    """
    clf = LogisticRegression(random_state=0, **kwargs)
    clf.fit(latents, condition)
    if get_accuracy:
        return clf.score(latents, condition)
    else:
        coefs = {}
        classes = clf.classes_.tolist()
        if len(classes) == 2:
            coefs[classes[0]] = np.ravel(clf.coef_)
        else:
            for i, cls in enumerate(classes):
                coefs[cls] = clf.coef_[i]
        return coefs
