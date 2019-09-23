#ifndef cluster_h
#define cluster_h

enum class Scheme {
  Kt = 1,
  CambridgeAachen = 0,
  AntiKt = -1
};

void cluster(PseudoJet *particles, int size, Scheme scheme, double r);

#endif  // cluster_h
