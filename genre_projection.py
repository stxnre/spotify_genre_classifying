import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('hw6/genre_train.csv')
y_train = train['label'].to_numpy()
X_train = train.drop('label',axis=1)

# PCA
pca = PCA(n_components = 2)
Xpca = pca.fit_transform(X_train)
tsne_decomp = pd.DataFrame({"P1":Xpca[:,0],"P2":Xpca[:,1]})
tsne_decomp['label'] = y_train

sns.scatterplot(data=tsne_decomp,x="P1",y="P2",s=8,hue="label",palette="bright")
plt.title("PCA of Spotify Training Data",fontsize=14)
plt.tight_layout()
plt.show()

# t-Distributed Stochastic Neighbor Embeddings
tsne = TSNE(n_components = 2, perplexity = 40, early_exaggeration = 12)
Xtsne= tsne.fit_transform(X_train)
tsne_decomp = pd.DataFrame({"P1":Xtsne[:,0],"P2":Xtsne[:,1]})
tsne_decomp['label'] = y_train

sns.scatterplot(data=tsne_decomp,x="P1",y="P2",s=8,hue="label",palette="bright")
plt.title("t-SNE on MNIST Training Data",fontsize=14)
plt.tight_layout()
plt.show()