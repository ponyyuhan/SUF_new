#pragma once

#include "suf/ir.hpp"
#include "suf/crypto/dpf.hpp"
#include "suf/pfss_batch.hpp"

#include <random>
#include <vector>

namespace suf {

struct PredQuery {
  int n_bits = 0;
  u64 threshold = 0;
  u64 input_add = 0;
  bool invert = false;
};

struct PfssPlan {
  std::vector<PredQuery> queries; // one per predicate in descriptor
};

inline PfssPlan compile_pfss_plan(const SUFDescriptor& d) {
  PfssPlan plan;
  plan.queries.reserve(d.predicates.size());
  for (const auto& p : d.predicates) {
    PredQuery q;
    switch (p.kind) {
      case PredKind::LT:
        q.n_bits = 64;
        q.threshold = p.param;
        break;
      case PredKind::LTLOW:
        q.n_bits = static_cast<int>(p.f);
        q.threshold = p.gamma;
        break;
      case PredKind::MSB:
        q.n_bits = 64;
        q.threshold = (1ULL << 63);
        q.invert = true;
        break;
      case PredKind::MSB_ADD:
        q.n_bits = 64;
        q.threshold = (1ULL << 63);
        q.input_add = p.param;
        q.invert = true;
        break;
      default:
        fail("unknown predicate kind in compile_pfss_plan");
    }
    plan.queries.push_back(q);
  }
  return plan;
}

inline DpfKeyBatch build_dpf_batch(const PfssPlan& plan, int party, std::mt19937_64& rng) {
  DpfKeyBatch batch;
  batch.n_bits = 0;
  batch.keys.reserve(plan.queries.size());

  for (const auto& q : plan.queries) {
    const int n_bits = q.n_bits;
    const u64 mask = (n_bits == 64) ? ~0ULL : ((1ULL << n_bits) - 1ULL);
    auto keys = keygen_dpf(n_bits, q.threshold, rng);
    const DpfKey& k = (party == 0) ? keys.first : keys.second;

    DpfKeyPacked packed;
    packed.root = k.root;
    packed.tcwL = k.tcwL;
    packed.tcwR = k.tcwR;
    packed.scw_offset = static_cast<int>(batch.scw.size());
    packed.n_bits = n_bits;
    packed.mask = mask;
    packed.t_init = static_cast<u8>(party & 1);
    packed.invert = (q.invert && party == 0) ? 1 : 0;
    packed.input_add = q.input_add;

    batch.keys.push_back(packed);
    batch.scw.insert(batch.scw.end(), k.scw.begin(), k.scw.end());
  }
  return batch;
}

} // namespace suf
