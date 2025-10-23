import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- Дані брендів (синтетичні, реалістичні діапазони) ----------
brands = [
    "L'Oréal","Sephora","Fenty Beauty","The Ordinary","Estée Lauder","NYX",
    "Maybelline","Clinique","Benefit","Huda Beauty","Lancôme","Shiseido",
    "Tarte","Glossier","Charlotte Tilbury"
]

def make_dataset(n=len(brands)):
    return pd.DataFrame({
        "Brand": brands[:n],
        "RevenueGrowth_%": np.clip(np.random.normal(18, 6, n), 5, 35),
        "OnlineShare_%": np.clip(np.random.normal(55, 15, n), 15, 95),
        "SubscriptionRate_%": np.clip(np.random.normal(7, 3, n), 0.5, 20),
        "AvgOrderValue_$": np.clip(np.random.normal(65, 20, n), 20, 150),
        "MarketingROI": np.clip(np.random.normal(2.4, 0.7, n), 0.6, 5),
        "NPS": np.clip(np.random.normal(48, 12, n), 10, 80),
        "Retention_%": np.clip(np.random.normal(42, 10, n), 15, 75),
        "MarketCoverage_%": np.clip(np.random.normal(50, 18, n), 10, 95),
    })

def minmax(df, cols):
    m, M = df[cols].min(), df[cols].max()
    return (df[cols] - m) / (M - m + 1e-12)

def composite_scores(df, w={"monet":0.4,"ux":0.35,"scal":0.25}):
    monet = ["RevenueGrowth_%","OnlineShare_%","SubscriptionRate_%","AvgOrderValue_$","MarketingROI"]
    ux    = ["NPS","Retention_%"]
    scal  = ["MarketCoverage_%","OnlineShare_%"]
    X = df.copy()
    Xn = minmax(X, list(set(monet+ux+scal)))
    X["Monetization"] = Xn[monet].mean(axis=1)
    X["UX"]           = Xn[ux].mean(axis=1)
    X["Scalability"]  = Xn[scal].mean(axis=1)
    X["Score"]        = w["monet"]*X["Monetization"] + w["ux"]*X["UX"] + w["scal"]*X["Scalability"]
    return X

# ---------- K-means (з нуля) ----------
def kmeans(X, k=3, max_iter=200, n_init=20):
    best = None; best_inertia = np.inf
    for _ in range(n_init):
        centers = X[np.random.choice(len(X), k, replace=False)]
        for _ in range(max_iter):
            d2 = ((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
            labels = d2.argmin(axis=1)
            new_centers = np.vstack([X[labels==j].mean(axis=0) if np.any(labels==j) else centers[j] for j in range(k)])
            if np.allclose(new_centers, centers): break
            centers = new_centers
        inertia = ((X-centers[labels])**2).sum()
        if inertia < best_inertia:
            best_inertia = inertia; best = (centers, labels)
    return best[0], best[1], best_inertia

# ---------- Спрощена PCA до 2D для візуалізації ----------
def pca2(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:,:2]*S[:2]

# ---------- SOM (карта Кохонена) ----------
class SOM:
    def __init__(self, m, n, dim, lr=0.5, steps=4000):
        self.m,self.n,self.dim,self.lr0,self.steps = m,n,dim,lr,steps
        self.W = np.random.normal(size=(m*n, dim))
        self.pos = np.array([[i,j] for i in range(m) for j in range(n)])
        self.radius0 = max(m,n)/2
    def _bmu(self, x):
        return ((self.W-x)**2).sum(axis=1).argmin()
    def fit(self, X):
        for t in range(self.steps):
            x = X[np.random.randint(len(X))]
            b = self._bmu(x)
            lr = self.lr0*np.exp(-t/self.steps)
            r  = self.radius0*np.exp(-t/self.steps)
            h  = np.exp(-((self.pos-self.pos[b])**2).sum(axis=1)/(2*r*r+1e-9))[:,None]
            self.W += lr*h*(x-self.W)
        return self
    def transform(self, X):
        return np.array([self._bmu(x) for x in X])

# ---------- ACO для TSP (маршрут доставки) ----------
def tsp_length(tour, D):
    return sum(D[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

def aco_tsp(coords, n_ants=25, iters=150, alpha=1.0, beta=3.0, rho=0.4, Q=50.0):
    n = len(coords)
    D = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(axis=2)) + 1e-12
    tau = np.ones((n,n))
    eta = 1.0 / D
    best_len = np.inf; best_tour = None
    rng = np.random.default_rng(0)
    for _ in range(iters):
        tours = []
        for _a in range(n_ants):
            unvisited = list(range(n))
            tour = [rng.integers(n)]; unvisited.remove(tour[0])
            while unvisited:
                i = tour[-1]
                p = np.array([(tau[i,j]**alpha)*(eta[i,j]**beta) for j in unvisited], float)
                p /= p.sum()
                j = unvisited[rng.choice(len(unvisited), p=p)]
                tour.append(j); unvisited.remove(j)
            tours.append(tour)
        tau *= (1 - rho)
        for tour in tours:
            L = tsp_length(tour, D)
            for i in range(n):
                a,b = tour[i], tour[(i+1)%n]
                tau[a,b] += Q / L; tau[b,a] += Q / L
            if L < best_len: best_len, best_tour = L, tour
    return best_tour, best_len

# ---------- Запуск платформи ----------
df = make_dataset()
X = composite_scores(df)
features = ["Monetization","UX","Scalability"]
Xmat = X[features].values

Z = pca2(Xmat)
centers, labels, inertia = kmeans(Xmat, k=3, n_init=30)
X["Cluster"] = labels

plt.figure(figsize=(6,5))
for c in np.unique(labels):
    idx = labels==c
    plt.scatter(Z[idx,0], Z[idx,1], s=70, label=f"Cluster {c}")
for i, name in enumerate(X["Brand"]):
    plt.text(Z[i,0]+0.02, Z[i,1]+0.02, name, fontsize=8)
plt.title("Позиціонування брендів (PCA + K-means)"); plt.legend(); plt.tight_layout(); plt.show()

top = X.sort_values("Score", ascending=False).reset_index(drop=True)
plt.figure(figsize=(8,5))
plt.barh(top["Brand"], top["Score"]); plt.gca().invert_yaxis()
plt.title("Інтегральний рейтинг брендів"); plt.xlabel("Score [0..1]"); plt.tight_layout(); plt.show()

som = SOM(8,8,dim=Xmat.shape[1], lr=0.4, steps=4000).fit(Xmat)
bmu = som.transform(Xmat)
plt.figure(figsize=(6,5))
plt.scatter(Z[:,0], Z[:,1], c=bmu, cmap="viridis", s=70)
for i,name in enumerate(X["Brand"]):
    plt.text(Z[i,0]+0.02, Z[i,1]+0.02, name, fontsize=8)
plt.title("SOM: топологічна карта брендів"); plt.tight_layout(); plt.show()

warehouses = np.random.uniform(0, 1, (15, 2))
best_tour, best_len = aco_tsp(warehouses)
plt.figure(figsize=(5,5))
plt.scatter(warehouses[:,0], warehouses[:,1], c='k')
route = best_tour + [best_tour[0]]
plt.plot(warehouses[route,0], warehouses[route,1], 'r-o')
plt.title(f"ACO: оптимальний маршрут доставки (довжина={best_len:.3f})")
plt.axis('equal'); plt.tight_layout(); plt.show()

print("\nТоп-5 брендів за Score:")
print(top[["Brand","Score","Monetization","UX","Scalability"]].head().to_string(index=False))
print("\nСередні профілі кластерів:")
print(X.groupby("Cluster")[features].mean().round(3).to_string())
