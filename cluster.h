#ifndef cluster_h
#define cluster_h

enum class Algorithm { Kt = 1, CambridgeAachen = 0, AntiKt = -1 };

void cluster(PseudoJet *particles, int size, Algorithm algo, double r);

#endif  // cluster_h
